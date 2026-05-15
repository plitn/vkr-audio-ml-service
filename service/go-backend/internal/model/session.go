package model

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

const (
	SessionStatusRecording         = "recording"
	SessionStatusRecordingFinished = "recording_finished"
	SessionStatusQueued            = "queued"
	SessionStatusProcessing        = "processing"
	SessionStatusDone              = "done"
	SessionStatusFailed            = "failed"
	ChunkStatusUploaded            = "uploaded"
	ChunkStatusQueued              = "queued"
	ChunkStatusProcessing          = "processing"
	ChunkStatusDone                = "done"
	ChunkStatusFailed              = "failed"
	DiarizationModeFull            = "full"
	DiarizationModeChunk           = "chunk"
)

type Session struct {
	ID                   uuid.UUID       `db:"id" json:"id"`
	UserID               uuid.UUID       `db:"user_id" json:"user_id"`
	Status               string          `db:"status" json:"status"`
	Language             string          `db:"language" json:"language"`
	Nr                   bool            `db:"nr" json:"nr"`
	Asr                  bool            `db:"asr" json:"asr"`
	Diar                 bool            `db:"diar" json:"diar"`
	DiarizationMode      string          `db:"diarization_mode" json:"diarization_mode"`
	ChunkDurationSec     int             `db:"chunk_duration_sec" json:"chunk_duration_sec"`
	TotalDurationSec     *float64        `db:"total_duration_sec" json:"total_duration_sec,omitempty"`
	FinalResult          json.RawMessage `db:"final_result" json:"final_result,omitempty"`
	Error                *string         `db:"error" json:"error,omitempty"`
	CreatedAt            time.Time       `db:"created_at" json:"created_at"`
	RecordingStartedAt   *time.Time      `db:"recording_started_at" json:"recording_started_at,omitempty"`
	RecordingFinishedAt  *time.Time      `db:"recording_finished_at" json:"recording_finished_at,omitempty"`
	ProcessingStartedAt  *time.Time      `db:"processing_started_at" json:"processing_started_at,omitempty"`
	ProcessingFinishedAt *time.Time      `db:"processing_finished_at" json:"processing_finished_at,omitempty"`
}

type Chunk struct {
	ID                   uuid.UUID       `db:"id" json:"id"`
	SessionID            uuid.UUID       `db:"session_id" json:"session_id"`
	ChunkIndex           int             `db:"chunk_index" json:"chunk_index"`
	Status               string          `db:"status" json:"status"`
	AudioKey             string          `db:"audio_key" json:"-"`
	NormalizedAudioKey   *string         `db:"normalized_audio_key" json:"normalized_audio_key,omitempty"`
	EnhancedAudioKey     *string         `db:"enhanced_audio_key" json:"enhanced_audio_key,omitempty"`
	StartSec             *float64        `db:"start_sec" json:"start_sec,omitempty"`
	DurationSec          *float64        `db:"duration_sec" json:"duration_sec,omitempty"`
	IsFinal              bool            `db:"is_final" json:"is_final"`
	Result               json.RawMessage `db:"result" json:"result,omitempty"`
	Error                *string         `db:"error" json:"error,omitempty"`
	CreatedAt            time.Time       `db:"created_at" json:"created_at"`
	ProcessingStartedAt  *time.Time      `db:"processing_started_at" json:"processing_started_at,omitempty"`
	ProcessingFinishedAt *time.Time      `db:"processing_finished_at" json:"processing_finished_at,omitempty"`
}

type ChunkResult struct {
	Transcript string        `json:"transcript,omitempty"`
	Segments   []interface{} `json:"segments,omitempty"`
	NrApplied  bool          `json:"nr_applied"`
	AsrApplied bool          `json:"asr_applied"`
}

type SSEEvent struct {
	Type       string      `json:"type"`
	ChunkIndex int         `json:"chunk_index"`
	Data       interface{} `json:"data"`
}

type ChunkMsgRedis struct {
	ChunkID         string `json:"chunk_id"`
	SessionID       string `json:"session_id"`
	AudioKey        string `json:"audio_key"`
	ChunkIndex      int    `json:"chunk_index"`
	IsFinal         bool   `json:"is_final"`
	Nr              bool   `json:"nr"`
	Asr             bool   `json:"asr"`
	Diar            bool   `json:"diar"`
	DiarizationMode string `json:"diarization_mode"`
	Language        string `json:"language"`
}

type SessionFinalizeMsgRedis struct {
	SessionID string `json:"session_id"`
}

type Tasks struct {
	Nr   bool `json:"nr"`
	Asr  bool `json:"asr"`
	Diar bool `json:"diar"`
}
