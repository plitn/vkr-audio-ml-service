package model

import "encoding/json"

type ErrorResponse struct {
	Error string `json:"error"`
}

type LoginResponse struct {
	Token string `json:"token"`
	User  *User  `json:"user"`
}

type SessionResultResponse struct {
	Session *Session `json:"session"`
	Chunks  []*Chunk `json:"chunks"`
}

type SpeakerLabelsRequest struct {
	SpeakerLabels map[string]string `json:"speaker_labels"`
}

type SpeakerLabelsResponse struct {
	FinalResult json.RawMessage `json:"final_result" swaggertype:"object"`
}
