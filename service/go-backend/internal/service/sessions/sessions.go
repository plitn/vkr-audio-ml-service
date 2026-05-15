// че по названию пакета
package service

import (
	"context"
	"encoding/json"
	"fmt"
	"go-backend/internal/model"
	"go-backend/internal/queue"
	"go-backend/internal/repository"
	"go-backend/internal/storage"
	"io"
	"mime/multipart"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
)

type SessionService struct {
	repo    repository.Repository
	queue   *queue.RedisQueue
	storage *storage.MinioStorage
}

func NewSessionService(
	repo repository.Repository,
	q *queue.RedisQueue,
	s *storage.MinioStorage,
) *SessionService {
	return &SessionService{repo: repo, queue: q, storage: s}
}

func (s *SessionService) CreateSession(
	ctx context.Context,
	userID uuid.UUID,
	nr, asr, diar bool,
	language string,
	diarizationMode string,
	chunkDurationSec int,
) (*model.Session, error) {
	if diarizationMode == "" {
		diarizationMode = model.DiarizationModeFull
	}
	if chunkDurationSec <= 0 {
		chunkDurationSec = 30
	}

	session := &model.Session{
		ID:               uuid.New(),
		UserID:           userID,
		Status:           model.SessionStatusRecording,
		Nr:               nr,
		Asr:              asr,
		Diar:             diar,
		Language:         language,
		DiarizationMode:  diarizationMode,
		ChunkDurationSec: chunkDurationSec,
	}

	if err := s.repo.CreateSession(ctx, session); err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	return session, nil
}

func (s *SessionService) AddChunk(ctx context.Context, sessionID uuid.UUID, chunkIndex int, isFinal bool,
	file multipart.File, header *multipart.FileHeader) (*model.Chunk, error) {

	session, err := s.repo.GetSessionByID(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	if session == nil {
		return nil, fmt.Errorf("session not found")
	}
	if session.Status != model.SessionStatusRecording {
		return nil, fmt.Errorf("session is not in recording state")
	}

	ext := strings.ToLower(filepath.Ext(header.Filename))
	if ext == "" {
		ext = ".wav"
	}
	audioKey := fmt.Sprintf("sessions/%s/chunk_%d%s", sessionID, chunkIndex, ext)

	contentType := header.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/octet-stream"
	}

	if err := s.storage.Upload(ctx, audioKey, file, header.Size, contentType); err != nil {
		return nil, fmt.Errorf("upload chunk: %w", err)
	}

	chunk := &model.Chunk{
		ID:         uuid.New(),
		SessionID:  sessionID,
		ChunkIndex: chunkIndex,
		AudioKey:   audioKey,
		Status:     model.ChunkStatusUploaded,
		IsFinal:    isFinal,
	}

	if err := s.repo.CreateChunk(ctx, chunk); err != nil {
		s.storage.Delete(ctx, audioKey)
		return nil, fmt.Errorf("create chunk: %w", err)
	}

	msg := model.ChunkMsgRedis{
		ChunkID:         chunk.ID.String(),
		SessionID:       sessionID.String(),
		AudioKey:        audioKey,
		ChunkIndex:      chunkIndex,
		IsFinal:         isFinal,
		Nr:              session.Nr,
		Asr:             session.Asr,
		Diar:            session.Diar,
		DiarizationMode: session.DiarizationMode,
		Language:        session.Language,
	}

	if err := s.queue.EnqueueChunk(ctx, msg); err != nil {
		return nil, fmt.Errorf("enqueue chunk: %w", err)
	}
	if err := s.repo.MarkChunkQueued(ctx, chunk.ID); err != nil {
		return nil, fmt.Errorf("mark chunk queued: %w", err)
	}
	chunk.Status = model.ChunkStatusQueued

	if isFinal {
		finalizeMsg := model.SessionFinalizeMsgRedis{
			SessionID: sessionID.String(),
		}
		if err := s.queue.EnqueueSessionFinalize(ctx, finalizeMsg); err != nil {
			return nil, fmt.Errorf("enqueue session finalize: %w", err)
		}
		if err := s.repo.MarkSessionRecordingFinished(ctx, sessionID); err != nil {
			return nil, fmt.Errorf("mark session recording finished: %w", err)
		}
	}

	return chunk, nil
}

func (s *SessionService) GetSession(ctx context.Context, id uuid.UUID) (*model.Session, error) {
	return s.repo.GetSessionByID(ctx, id)
}

func (s *SessionService) GetUserSessions(ctx context.Context, userID uuid.UUID) ([]*model.Session, error) {
	return s.repo.GetSessionsByUserID(ctx, userID)
}

func (s *SessionService) GetResult(ctx context.Context, sessionID uuid.UUID) ([]*model.Chunk, error) {
	return s.repo.GetChunksBySession(ctx, sessionID)
}

func (s *SessionService) GetDoneChunksSince(ctx context.Context, sessionID uuid.UUID, fromIndex int) ([]*model.Chunk, error) {
	return s.repo.GetDoneChunksSince(ctx, sessionID, fromIndex)
}

func (s *SessionService) DownloadArtifact(ctx context.Context, key string) (io.ReadCloser, error) {
	return s.storage.Download(ctx, key)
}

func (s *SessionService) UpdateSpeakerLabels(ctx context.Context, session *model.Session, labels map[string]string) (json.RawMessage, error) {
	if len(session.FinalResult) == 0 || string(session.FinalResult) == "null" {
		return nil, fmt.Errorf("final result is not available")
	}

	var result map[string]interface{}
	if err := json.Unmarshal(session.FinalResult, &result); err != nil {
		return nil, fmt.Errorf("parse final result: %w", err)
	}

	cleanLabels := make(map[string]string)
	for speaker, label := range labels {
		speaker = strings.TrimSpace(speaker)
		label = strings.TrimSpace(label)
		if speaker == "" || label == "" {
			continue
		}
		cleanLabels[speaker] = label
	}

	result["speaker_labels"] = cleanLabels
	applySpeakerLabels(result, cleanLabels)

	updated, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("encode final result: %w", err)
	}
	if err := s.repo.UpdateSessionFinalResult(ctx, session.ID, updated); err != nil {
		return nil, fmt.Errorf("update final result: %w", err)
	}

	return updated, nil
}

func applySpeakerLabels(result map[string]interface{}, labels map[string]string) {
	applyLabelsToItems(result["segments"], labels)
	applyLabelsToItems(result["speaker_turns"], labels)
}

func applyLabelsToItems(value interface{}, labels map[string]string) {
	items, ok := value.([]interface{})
	if !ok {
		return
	}
	for _, item := range items {
		object, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		speaker, ok := object["speaker"].(string)
		if !ok || speaker == "" {
			continue
		}
		if label, exists := labels[speaker]; exists {
			object["speaker_label"] = label
		} else {
			delete(object, "speaker_label")
		}
	}
}
