package jobs

import (
	"context"
	"fmt"
	"go-backend/internal/model"
	"go-backend/internal/queue"
	"go-backend/internal/repository"
	"go-backend/internal/storage"
	"mime/multipart"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
)

type Jobs struct {
	repo    repository.Repository
	storage storage.MinioStorage
	redis   queue.RedisQueue
}

func NewJobs(repo repository.Repository, minio storage.MinioStorage, queue queue.RedisQueue) *Jobs {
	return &Jobs{repo: repo, storage: minio, redis: queue}
}

func (s *Jobs) CreateJob(ctx context.Context, file multipart.File, header *multipart.FileHeader,
	nr, asr, diar bool, language string, userId uuid.UUID) (model.Job, error) {
	jobID := uuid.New()
	ext := strings.ToLower(filepath.Ext(header.Filename))
	audioKey := jobID.String() + ext

	contentType := map[string]string{
		".wav":  "audio/wav",
		".mp3":  "audio/mpeg",
		".ogg":  "audio/ogg",
		".flac": "audio/flac",
		".m4a":  "audio/mp4",
	}[ext]

	if err := s.storage.Upload(ctx, audioKey, file, header.Size, contentType); err != nil {
		return model.Job{}, fmt.Errorf("upload audio: %w", err)
	}

	job := model.Job{
		ID:       jobID,
		Nr:       nr,
		Asr:      asr,
		Diar:     diar,
		AudioKey: audioKey,
		Language: language,
		UserID:   userId,
	}

	if err := s.repo.CreateJob(ctx, job); err != nil {
		s.storage.Delete(ctx, audioKey)
		return model.Job{}, fmt.Errorf("create job: %w", err)
	}

	msg := model.JobMsgRedis{
		JobId:    jobID.String(),
		AudioKey: audioKey,
		NR:       nr,
		Asr:      asr,
		Diar:     diar,
		Language: language,
	}

	if err := s.redis.Enqueue(ctx, msg); err != nil {
		s.repo.SetJobStatus(ctx, jobID, "failed to enqueue: "+err.Error())
		return model.Job{}, fmt.Errorf("enqueue: %w", err)
	}

	return job, nil
}

func (s *Jobs) GetJob(ctx context.Context, id uuid.UUID) (model.Job, error) {
	return s.repo.GetByID(ctx, id)
}

func (s *Jobs) SetJobStatus(ctx context.Context, id uuid.UUID, status string) error {
	return s.repo.SetJobStatus(ctx, id, status)
}

func (s *Jobs) DeleteJob(ctx context.Context, id uuid.UUID) error {
	return s.repo.DeleteJob(ctx, id)
}

func (s *Jobs) GetJobsByUser(ctx context.Context, userID uuid.UUID) ([]model.Job, error) {
	return s.repo.GetJobsByUserID(ctx, userID)
}
