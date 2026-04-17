package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"go-backend/internal/model"

	"github.com/redis/go-redis/v9"
)

const jobQueueKey = "jobs:queue"

type RedisQueue struct {
	client *redis.Client
}

func NewRedisQueue(client *redis.Client) *RedisQueue {
	return &RedisQueue{client: client}
}

func (q *RedisQueue) Enqueue(ctx context.Context, msg model.JobMsgRedis) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal message: %w", err)
	}

	if err := q.client.LPush(ctx, jobQueueKey, data).Err(); err != nil {
		return fmt.Errorf("lpush: %w", err)
	}

	return nil
}
