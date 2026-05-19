package repository

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"go-backend/internal/model"

	"github.com/google/uuid"
)

func (r *repository) CreateSession(ctx context.Context, s *model.Session) error {
	query := `INSERT INTO sessions (id, user_id, status, language, nr, asr, diar, diarization_mode, expected_speakers, chunk_duration_sec)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`
	_, err := r.db.ExecContext(ctx, query,
		s.ID, s.UserID, s.Status, s.Language, s.Nr, s.Asr, s.Diar, s.DiarizationMode, s.ExpectedSpeakers, s.ChunkDurationSec)
	if err != nil {
		return fmt.Errorf("create session: %w", err)
	}
	return nil
}

func (r *repository) GetSessionByID(ctx context.Context, id uuid.UUID) (*model.Session, error) {
	query := `SELECT id, user_id, status, language, nr, asr, diar, diarization_mode, expected_speakers,
			chunk_duration_sec, total_duration_sec, COALESCE(final_result, 'null'::jsonb) AS final_result, error,
			created_at, recording_started_at, recording_finished_at,
			processing_started_at, processing_finished_at FROM sessions WHERE id = $1`
	var s model.Session
	err := r.db.GetContext(ctx, &s, query, id)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get session: %w", err)
	}
	return &s, nil
}

func (r *repository) GetSessionsByUserID(ctx context.Context, userID uuid.UUID) ([]*model.Session, error) {
	query := `SELECT id, user_id, status, language, nr, asr, diar, diarization_mode, expected_speakers,
			chunk_duration_sec, total_duration_sec, COALESCE(final_result, 'null'::jsonb) AS final_result, error,
			created_at, recording_started_at, recording_finished_at,
			processing_started_at, processing_finished_at FROM sessions WHERE user_id = $1 ORDER BY created_at DESC`
	var sessions []*model.Session
	err := r.db.SelectContext(ctx, &sessions, query, userID)
	if err != nil {
		return nil, fmt.Errorf("get sessions by user: %w", err)
	}
	return sessions, nil
}

func (r *repository) MarkSessionRecordingFinished(ctx context.Context, id uuid.UUID) error {
	query := `UPDATE sessions SET status = $1, recording_finished_at = now() WHERE id = $2`
	_, err := r.db.ExecContext(ctx, query, model.SessionStatusRecordingFinished, id)
	return err
}

func (r *repository) MarkSessionProcessing(ctx context.Context, id uuid.UUID) error {
	query := `UPDATE sessions SET status = $1, processing_started_at = COALESCE(processing_started_at, now()), error = NULL WHERE id = $2`
	_, err := r.db.ExecContext(ctx, query, model.SessionStatusProcessing, id)
	return err
}

func (r *repository) MarkSessionDone(ctx context.Context, id uuid.UUID, totalDurationSec *float64, finalResult []byte) error {
	var finalResultValue interface{}
	if finalResult != nil {
		finalResultValue = string(finalResult)
	}

	query := `UPDATE sessions SET status = $1, total_duration_sec = $2, final_result = $3::jsonb, error = NULL, processing_finished_at = now() WHERE id = $4`
	_, err := r.db.ExecContext(ctx, query, model.SessionStatusDone, totalDurationSec, finalResultValue, id)
	return err
}

func (r *repository) UpdateSessionFinalResult(ctx context.Context, id uuid.UUID, finalResult []byte) error {
	query := `UPDATE sessions SET final_result = $1::jsonb WHERE id = $2`
	_, err := r.db.ExecContext(ctx, query, string(finalResult), id)
	return err
}

func (r *repository) MarkSessionFailed(ctx context.Context, id uuid.UUID, errorMessage string) error {
	query := `UPDATE sessions SET status = $1, error = $2, processing_finished_at = now() WHERE id = $3`
	_, err := r.db.ExecContext(ctx, query, model.SessionStatusFailed, errorMessage, id)
	return err
}

func (r *repository) CreateChunk(ctx context.Context, c *model.Chunk) error {
	query := `INSERT INTO chunks (id, session_id, chunk_index, audio_key, status, is_final) VALUES ($1, $2, $3, $4, $5, $6)`
	_, err := r.db.ExecContext(ctx, query, c.ID, c.SessionID, c.ChunkIndex, c.AudioKey, c.Status, c.IsFinal)
	if err != nil {
		return fmt.Errorf("create chunk: %w", err)
	}
	return nil
}

func (r *repository) MarkChunkQueued(ctx context.Context, id uuid.UUID) error {
	query := `UPDATE chunks SET status = $1, error = NULL WHERE id = $2`
	_, err := r.db.ExecContext(ctx, query, model.ChunkStatusQueued, id)
	return err
}

func (r *repository) MarkChunkProcessing(ctx context.Context, id uuid.UUID) error {
	query := `UPDATE chunks SET status = $1, processing_started_at = COALESCE(processing_started_at, now()), error = NULL WHERE id = $2 AND status IN ($3, $4)`
	_, err := r.db.ExecContext(ctx, query, model.ChunkStatusProcessing, id, model.ChunkStatusUploaded, model.ChunkStatusQueued)
	return err
}

func (r *repository) MarkChunkDone(ctx context.Context, id uuid.UUID, normalizedAudioKey, enhancedAudioKey *string, startSec, durationSec *float64, result []byte) error {
	var resultValue interface{}
	if result != nil {
		resultValue = string(result)
	}

	query := `UPDATE chunks SET status = $1, normalized_audio_key = $2, enhanced_audio_key = $3, start_sec = $4, 
                  duration_sec = $5, result = $6::jsonb, error = NULL, processing_finished_at = now() WHERE id = $7
	`
	_, err := r.db.ExecContext(ctx, query, model.ChunkStatusDone, normalizedAudioKey, enhancedAudioKey, startSec, durationSec, resultValue, id)
	return err
}

func (r *repository) MarkChunkFailed(ctx context.Context, id uuid.UUID, errorMessage string) error {
	query := `UPDATE chunks SET status = $1, error = $2, processing_finished_at = now() WHERE id = $3`
	_, err := r.db.ExecContext(ctx, query, model.ChunkStatusFailed, errorMessage, id)
	return err
}

func (r *repository) GetChunksBySession(ctx context.Context, sessionID uuid.UUID) ([]*model.Chunk, error) {
	query := `SELECT id, session_id, chunk_index, status, audio_key, normalized_audio_key,
       enhanced_audio_key, start_sec, duration_sec, is_final, COALESCE(result, 'null'::jsonb) AS result, error,
       created_at, processing_started_at, processing_finished_at FROM chunks WHERE session_id = $1 ORDER BY chunk_index ASC`
	var chunks []*model.Chunk
	err := r.db.SelectContext(ctx, &chunks, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("get chunks: %w", err)
	}
	return chunks, nil
}

func (r *repository) GetDoneChunksSince(ctx context.Context, sessionID uuid.UUID, fromIndex int) ([]*model.Chunk, error) {
	query := `SELECT id, session_id, chunk_index, status, audio_key, normalized_audio_key,
       enhanced_audio_key, start_sec, duration_sec, is_final, COALESCE(result, 'null'::jsonb) AS result, error,
       created_at, processing_started_at, processing_finished_at 
	FROM chunks WHERE session_id = $1 AND chunk_index >= $2 AND status = 'done' ORDER BY chunk_index ASC
	`
	var chunks []*model.Chunk
	err := r.db.SelectContext(ctx, &chunks, query, sessionID, fromIndex)
	if err != nil {
		return nil, fmt.Errorf("get chunks: %w", err)
	}
	return chunks, nil
}
