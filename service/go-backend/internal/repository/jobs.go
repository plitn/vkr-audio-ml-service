package repository

import (
	"context"
	"fmt"
	"go-backend/internal/model"
	"go-backend/internal/status"

	"github.com/google/uuid"
)

func (r *repository) CreateJob(ctx context.Context, job model.Job) error {
	query := `INSERT INTO jobs (id, status, nr, asr, diar, audio_key, language, user_id) 
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`

	_, err := r.db.ExecContext(ctx, query, job.ID, status.Pending, job.Nr, job.Asr, job.Diar, job.AudioKey, job.Language, job.UserID)
	if err != nil {
		return fmt.Errorf("insert job: %w", err)
	}
	return nil
}

func (r *repository) GetByID(ctx context.Context, id uuid.UUID) (model.Job, error) {
	var jobResult model.Job
	query := `SELECT id, status, nr, asr, diar, audio_key, language, result, error, created_at, finished_at FROM jobs
		WHERE id = $1`

	err := r.db.GetContext(ctx, &jobResult, query, id)
	if err != nil {
		return model.Job{}, fmt.Errorf("get job: %w", err)
	}

	return jobResult, nil
}

func (r *repository) SetJobStatus(ctx context.Context, id uuid.UUID, status string) error {
	query := `UPDATE jobs SET status = $1 WHERE id = $2`
	_, err := r.db.ExecContext(ctx, query, status, id)
	if err != nil {
		return fmt.Errorf("set running: %w", err)
	}
	return nil
}

func (r *repository) DeleteJob(ctx context.Context, id uuid.UUID) error {
	query := `UPDATE jobs SET status = 'deleted' WHERE id = $1`
	_, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("set running: %w", err)
	}
	return nil
}
