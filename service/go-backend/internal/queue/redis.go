package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"go-backend/internal/model"

	"github.com/redis/go-redis/v9"
)

const (
	chunkQueueKey           = "chunks:queue"
	sessionFinalizeQueueKey = "sessions:finalize:queue"
)

type RedisQueue struct {
	client *redis.Client
}

func NewRedisQueue(client *redis.Client) *RedisQueue {
	return &RedisQueue{client: client}
}

func (q *RedisQueue) EnqueueChunk(ctx context.Context, msg model.ChunkMsgRedis) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal chunk message: %w", err)
	}
	return q.client.LPush(ctx, chunkQueueKey, data).Err()
}

func (q *RedisQueue) EnqueueSessionFinalize(ctx context.Context, msg model.SessionFinalizeMsgRedis) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal session finalize message: %w", err)
	}
	return q.client.LPush(ctx, sessionFinalizeQueueKey, data).Err()
}
