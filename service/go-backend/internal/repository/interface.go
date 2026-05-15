package repository

import (
	"context"
	"go-backend/internal/model"

	"github.com/google/uuid"
)

type Repository interface {
	JobRepository
	UserRepository
	SessionRepository
}

type JobRepository interface {
	CreateJob(ctx context.Context, job model.Job) error
	GetByID(ctx context.Context, id uuid.UUID) (model.Job, error)
	DeleteJob(ctx context.Context, id uuid.UUID) error
	SetJobStatus(ctx context.Context, id uuid.UUID, status string) error
	GetJobsByUserID(ctx context.Context, userID uuid.UUID) ([]model.Job, error)
}

type UserRepository interface {
	CreateUser(ctx context.Context, user model.User) error
	GetUserByEmail(ctx context.Context, email string) (model.User, error)
	GetUserByID(ctx context.Context, id uuid.UUID) (model.User, error)
}

type SessionRepository interface {
	CreateSession(ctx context.Context, s *model.Session) error
	GetSessionByID(ctx context.Context, id uuid.UUID) (*model.Session, error)
	GetSessionsByUserID(ctx context.Context, userID uuid.UUID) ([]*model.Session, error)
	MarkSessionRecordingFinished(ctx context.Context, id uuid.UUID) error
	MarkSessionProcessing(ctx context.Context, id uuid.UUID) error
	MarkSessionDone(ctx context.Context, id uuid.UUID, totalDurationSec *float64, finalResult []byte) error
	UpdateSessionFinalResult(ctx context.Context, id uuid.UUID, finalResult []byte) error
	MarkSessionFailed(ctx context.Context, id uuid.UUID, errorMessage string) error
	CreateChunk(ctx context.Context, c *model.Chunk) error
	MarkChunkQueued(ctx context.Context, id uuid.UUID) error
	MarkChunkProcessing(ctx context.Context, id uuid.UUID) error
	MarkChunkDone(ctx context.Context, id uuid.UUID, normalizedAudioKey, enhancedAudioKey *string, startSec, durationSec *float64, result []byte) error
	MarkChunkFailed(ctx context.Context, id uuid.UUID, errorMessage string) error
	GetChunksBySession(ctx context.Context, sessionID uuid.UUID) ([]*model.Chunk, error)
	GetDoneChunksSince(ctx context.Context, sessionID uuid.UUID, fromIndex int) ([]*model.Chunk, error)
}
