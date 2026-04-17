package handler

import "net/http"

type JobsHandler interface {
	GetJob(w http.ResponseWriter, r *http.Request)
}
