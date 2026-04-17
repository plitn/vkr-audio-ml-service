package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/jmoiron/sqlx"
	"github.com/redis/go-redis/v9"
)

type HealthHandler struct {
	db  *sqlx.DB
	rdb *redis.Client
}

func NewHealthHandler(db *sqlx.DB, rdb *redis.Client) *HealthHandler {
	return &HealthHandler{db: db, rdb: rdb}
}

func (h *HealthHandler) Health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	status := map[string]string{
		"status":   "ok",
		"postgres": "ok",
		"redis":    "ok",
	}
	code := http.StatusOK

	if err := h.db.PingContext(ctx); err != nil {
		status["status"] = "degraded"
		status["postgres"] = "unavailable"
		code = http.StatusServiceUnavailable
	}

	if err := h.rdb.Ping(ctx).Err(); err != nil {
		status["status"] = "degraded"
		status["redis"] = "unavailable"
		code = http.StatusServiceUnavailable
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(status)
}
