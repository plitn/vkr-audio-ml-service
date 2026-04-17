package model

import (
	"database/sql"

	"github.com/google/uuid"
)

type TaskConf struct {
	NR   bool `json:"nr"`
	Asr  bool `json:"asr"`
	Diar bool `json:"diar"`
}

type Job struct {
	ID         uuid.UUID    `db:"id" json:"id"`
	Status     string       `db:"status" json:"status"`
	Nr         bool         `db:"nr" json:"nr"`
	Asr        bool         `db:"asr" json:"asr"`
	Diar       bool         `db:"diar" json:"diar"`
	AudioKey   string       `db:"audio_key"`
	Language   string       `db:"language" json:"language"`
	Result     *string      `db:"result" json:"result"`
	Error      *string      `db:"error" json:"error"`
	CreatedAt  sql.NullTime `db:"created_at" json:"created_at"`
	FinishedAt sql.NullTime `db:"finished_at" json:"finished_at"`
	UserID     uuid.UUID    `db:"user_id" json:"user_id"`
}

type JobMsgRedis struct {
	JobId    string `json:"job_id"`
	AudioKey string `db:"audio_key" json:"audio_key"`
	NR       bool   `db:"nr" json:"nr"`
	Asr      bool   `db:"asr" json:"asr"`
	Diar     bool   `db:"diar" json:"diar"`
	Language string `json:"language"`
}
