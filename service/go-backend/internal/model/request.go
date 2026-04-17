package model

type СonfigPayload struct {
	Tasks    TaskConf `json:"tasks"`
	Language string   `json:"language"`
}

type StatusBody struct {
	Status string `json:"status"`
}
