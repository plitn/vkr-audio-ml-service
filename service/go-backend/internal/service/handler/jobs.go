package handler

import (
	"encoding/json"
	"fmt"
	"go-backend/internal/middleware"
	"go-backend/internal/model"
	"go-backend/internal/service/jobs"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/go-chi/chi"
	"github.com/google/uuid"
)

type jobsHandler struct {
	jobsService *jobs.Jobs
}

func NewJobsHandler(jobs *jobs.Jobs) *jobsHandler {
	return &jobsHandler{
		jobsService: jobs,
	}
}

func (h *jobsHandler) GetJob(w http.ResponseWriter, r *http.Request) {
	rawID := chi.URLParam(r, "id")
	//rawID := r.PathValue("id")

	id, err := uuid.Parse(rawID)
	if err != nil {
		fmt.Println(err)
		writeError(w, http.StatusBadRequest, "invalid job id")
		return
	}

	job, err := h.jobsService.GetJob(r.Context(), id)
	if err != nil {
		fmt.Println(err)
		writeError(w, http.StatusInternalServerError, "failed to get job")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(job)
}

func (h *jobsHandler) CreateJob(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 100<<20)

	if err := r.ParseMultipartForm(100 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "failed to parse form")
		return
	}

	configStr := r.FormValue("config")
	if configStr == "" {
		writeError(w, http.StatusBadRequest, "config field is required")
		return
	}

	var cfg model.СonfigPayload
	if err := json.Unmarshal([]byte(configStr), &cfg); err != nil {
		writeError(w, http.StatusBadRequest, "invalid config JSON")
		return
	}

	if !cfg.Tasks.NR && !cfg.Tasks.Asr && !cfg.Tasks.Diar {
		writeError(w, http.StatusBadRequest, "at least one task must be enabled")
		return
	}

	language := cfg.Language
	if language == "" {
		language = "english"
	}

	file, header, err := r.FormFile("audio")
	if err != nil {
		writeError(w, http.StatusBadRequest, "audio file is required")
		return
	}
	defer file.Close()

	if !checkFileExt(filepath.Ext(header.Filename)) {
		writeError(w, http.StatusBadRequest, "unsupported format: "+filepath.Ext(header.Filename))
		return
	}
	userID, ok := r.Context().Value(middleware.UserIDKey).(uuid.UUID)
	if !ok {
		fmt.Printf("user id: %v\n", userID)
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	job, err := h.jobsService.CreateJob(r.Context(), file, header, cfg.Tasks.NR, cfg.Tasks.Asr, cfg.Tasks.Diar, language, userID)
	if err != nil {
		fmt.Println(err)
		writeError(w, http.StatusInternalServerError, "failed to create job")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(job)
}

func (h *jobsHandler) SetJobStatus(w http.ResponseWriter, r *http.Request) {
	rawID := chi.URLParam(r, "id")

	id, err := uuid.Parse(rawID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid job id")
		return
	}

	var body model.StatusBody

	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if !checkJobStatus(body.Status) {
		writeError(w, http.StatusBadRequest, "invalid status: "+body.Status)
		return
	}

	if err := h.jobsService.SetJobStatus(r.Context(), id, body.Status); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to update status")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": body.Status})
}

func (h *jobsHandler) DeleteJob(w http.ResponseWriter, r *http.Request) {
	rawID := chi.URLParam(r, "id")

	id, err := uuid.Parse(rawID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid job id")
		return
	}

	if err := h.jobsService.DeleteJob(r.Context(), id); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to delete job")
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (h *jobsHandler) GetUserJobs(w http.ResponseWriter, r *http.Request) {
	rawID := chi.URLParam(r, "user_id")

	userID, err := uuid.Parse(rawID)
	if err != nil {
		fmt.Println(err)
		writeError(w, http.StatusBadRequest, "invalid user_id")
		return
	}

	userJobs, err := h.jobsService.GetJobsByUser(r.Context(), userID)
	if err != nil {
		fmt.Println(err)
		writeError(w, http.StatusInternalServerError, "failed to get jobs")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(userJobs)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func checkFileExt(ext string) bool {
	ext = strings.ToLower(ext)
	allowed := map[string]bool{
		".wav":  true,
		".mp3":  true,
		".ogg":  true,
		".flac": true,
		".m4a":  true,
	}
	return allowed[ext]
}

func checkJobStatus(status string) bool {
	allowed := map[string]bool{
		"pending": true,
		"running": true,
		"done":    true,
		"failed":  true,
	}
	return allowed[status]
}
