package repository

import (
	"context"
	"go-backend/internal/model"

	"github.com/google/uuid"
)

type Repository interface {
	JobRepository
	UserRepository
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
