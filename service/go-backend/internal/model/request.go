package model

type СonfigPayload struct {
	Tasks    TaskConf `json:"tasks"`
	Language string   `json:"language"`
}

type StatusBody struct {
	Status string `json:"status"`
}

type TaskBody struct {
	Tasks            Tasks  `json:"tasks"`
	Language         string `json:"language"`
	DiarizationMode  string `json:"diarization_mode"`
	ChunkDurationSec int    `json:"chunk_duration_sec"`
}
