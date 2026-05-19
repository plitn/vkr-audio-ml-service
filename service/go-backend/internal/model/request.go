package model

type TaskBody struct {
	Tasks            Tasks  `json:"tasks"`
	Language         string `json:"language"`
	DiarizationMode  string `json:"diarization_mode"`
	ExpectedSpeakers *int   `json:"expected_speakers,omitempty"`
	ChunkDurationSec int    `json:"chunk_duration_sec"`
}
