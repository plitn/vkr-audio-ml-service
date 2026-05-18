package repository

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"go-backend/internal/model"

	"github.com/google/uuid"
)

func (r *repository) CreateUser(ctx context.Context, user model.User) error {
	query := `INSERT INTO users (id, email, password_hash) VALUES ($1, $2, $3)`
	_, err := r.db.ExecContext(ctx, query, user.ID, user.Email, user.PasswordHash)
	if err != nil {
		return fmt.Errorf("create user: %w", err)
	}
	return nil
}

func (r *repository) GetUserByEmail(ctx context.Context, email string) (model.User, error) {
	query := `SELECT id, email, password_hash FROM users WHERE email = $1`
	var user model.User
	err := r.db.GetContext(ctx, &user, query, email)
	if errors.Is(err, sql.ErrNoRows) {
		return user, nil
	}
	if err != nil {
		return user, fmt.Errorf("get user by email: %w", err)
	}
	return user, nil
}

func (r *repository) GetUserByID(ctx context.Context, id uuid.UUID) (model.User, error) {
	query := `SELECT id, email, password_hash, created_at FROM users WHERE id = $1`
	var user model.User
	err := r.db.GetContext(ctx, &user, query, id)
	if errors.Is(err, sql.ErrNoRows) {
		return user, nil
	}
	if err != nil {
		return user, fmt.Errorf("get user by id: %w", err)
	}
	return user, nil
}
