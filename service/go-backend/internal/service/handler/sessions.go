package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/go-chi/chi"
	"go-backend/internal/middleware"
	"go-backend/internal/model"
	service "go-backend/internal/service/sessions"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
)

type SessionHandler struct {
	sessionService *service.SessionService
}

func NewSessionHandler(s *service.SessionService) *SessionHandler {
	return &SessionHandler{sessionService: s}
}

// ListSessions godoc
// @Summary List recording sessions
// @Description Returns previous recording sessions for the authenticated user.
// @Tags sessions
// @Produce json
// @Security BearerAuth
// @Success 200 {array} model.Session
// @Failure 401 {object} model.ErrorResponse
// @Failure 500 {object} model.ErrorResponse
// @Router /api/v1/sessions [get]
func (h *SessionHandler) ListSessions(w http.ResponseWriter, r *http.Request) {
	userID, ok := r.Context().Value(middleware.UserIDKey).(uuid.UUID)
	if !ok {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	sessions, err := h.sessionService.GetUserSessions(r.Context(), userID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to get sessions")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(sessions)
}

// CreateSession godoc
// @Summary Create a recording session
// @Description Creates a new session with selected ML tasks and diarization mode.
// @Tags sessions
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param request body model.TaskBody true "Session task configuration"
// @Success 201 {object} model.Session
// @Failure 400 {object} model.ErrorResponse
// @Failure 401 {object} model.ErrorResponse
// @Failure 500 {object} model.ErrorResponse
// @Router /api/v1/sessions [post]
func (h *SessionHandler) CreateSession(w http.ResponseWriter, r *http.Request) {
	userID, ok := r.Context().Value(middleware.UserIDKey).(uuid.UUID)
	if !ok {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	var body model.TaskBody

	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if !body.Tasks.Nr && !body.Tasks.Asr && !body.Tasks.Diar {
		writeError(w, http.StatusBadRequest, "at least one task must be enabled")
		return
	}
	if body.Tasks.Diar && !body.Tasks.Asr {
		writeError(w, http.StatusBadRequest, "diarization requires ASR")
		return
	}
	if body.ExpectedSpeakers != nil && !body.Tasks.Diar {
		writeError(w, http.StatusBadRequest, "expected_speakers requires diarization")
		return
	}
	if body.ExpectedSpeakers != nil && (*body.ExpectedSpeakers < 1 || *body.ExpectedSpeakers > 20) {
		writeError(w, http.StatusBadRequest, "expected_speakers must be between 1 and 20")
		return
	}

	language := body.Language
	if language == "" {
		language = "auto"
	}
	diarizationMode := body.DiarizationMode
	if diarizationMode == "" {
		diarizationMode = model.DiarizationModeFull
	}
	if diarizationMode != model.DiarizationModeFull && diarizationMode != model.DiarizationModeChunk {
		writeError(w, http.StatusBadRequest, "invalid diarization_mode")
		return
	}

	session, err := h.sessionService.CreateSession(
		r.Context(), userID,
		body.Tasks.Nr, body.Tasks.Asr, body.Tasks.Diar,
		language,
		diarizationMode,
		body.ExpectedSpeakers,
		body.ChunkDurationSec,
	)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create session")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(session)
}

// AddChunk godoc
// @Summary Upload an audio chunk
// @Description Uploads one audio chunk for a session, stores it, and enqueues async worker processing.
// @Tags sessions
// @Accept multipart/form-data
// @Produce json
// @Security BearerAuth
// @Param id path string true "Session ID"
// @Param audio formData file true "Audio chunk file"
// @Param chunk_index formData int true "Zero-based chunk index"
// @Param is_final formData bool true "Marks the last chunk of the recording"
// @Success 202 {object} model.Chunk
// @Failure 400 {object} model.ErrorResponse
// @Failure 401 {object} model.ErrorResponse
// @Failure 404 {object} model.ErrorResponse
// @Failure 409 {object} model.ErrorResponse
// @Failure 500 {object} model.ErrorResponse
// @Router /api/v1/sessions/{id}/chunks [post]
func (h *SessionHandler) AddChunk(w http.ResponseWriter, r *http.Request) {
	rawID := chi.URLParam(r, "id")
	sessionID, err := uuid.Parse(rawID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB на чанк
	if err := r.ParseMultipartForm(10 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "failed to parse form")
		return
	}

	chunkIndexStr := r.FormValue("chunk_index")
	chunkIndex, err := strconv.Atoi(chunkIndexStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid chunk_index")
		return
	}

	isFinal := r.FormValue("is_final") == "true"

	file, header, err := r.FormFile("audio")
	if err != nil {
		writeError(w, http.StatusBadRequest, "audio file is required")
		return
	}
	defer file.Close()

	chunk, err := h.sessionService.AddChunk(r.Context(), sessionID, chunkIndex, isFinal, file, header)
	if err != nil {
		if err.Error() == "session not found" {
			writeError(w, http.StatusNotFound, "session not found")
			return
		}
		if err.Error() == "session is not in recording state" {
			writeError(w, http.StatusConflict, "session is already finished")
			return
		}
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to add chunk: %s", err.Error()))
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(chunk)
}

// GetResult godoc
// @Summary Get session result
// @Description Returns the session state and chunks. The final transcript is available when session.status is done.
// @Tags sessions
// @Produce json
// @Security BearerAuth
// @Param id path string true "Session ID"
// @Success 200 {object} model.SessionResultResponse
// @Failure 400 {object} model.ErrorResponse
// @Failure 401 {object} model.ErrorResponse
// @Failure 404 {object} model.ErrorResponse
// @Failure 500 {object} model.ErrorResponse
// @Router /api/v1/sessions/{id}/result [get]
func (h *SessionHandler) GetResult(w http.ResponseWriter, r *http.Request) {
	userID, ok := r.Context().Value(middleware.UserIDKey).(uuid.UUID)
	if !ok {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	rawID := chi.URLParam(r, "id")
	sessionID, err := uuid.Parse(rawID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	session, err := h.sessionService.GetSession(r.Context(), sessionID)
	if err != nil || session == nil {
		writeError(w, http.StatusNotFound, "session not found")
		return
	}
	if session.UserID != userID {
		writeError(w, http.StatusNotFound, "session not found")
		return
	}

	chunks, err := h.sessionService.GetResult(r.Context(), sessionID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to get result")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"session": session,
		"chunks":  chunks,
	})
}

// UpdateSpeakerLabels godoc
// @Summary Update speaker labels
// @Description Saves user-provided speaker names and applies them to the final result.
// @Tags sessions
// @Accept json
// @Produce json
// @Security BearerAuth
// @Param id path string true "Session ID"
// @Param request body model.SpeakerLabelsRequest true "Speaker label mapping"
// @Success 200 {object} model.SpeakerLabelsResponse
// @Failure 400 {object} model.ErrorResponse
// @Failure 401 {object} model.ErrorResponse
// @Failure 404 {object} model.ErrorResponse
// @Failure 409 {object} model.ErrorResponse
// @Router /api/v1/sessions/{id}/speaker-labels [patch]
func (h *SessionHandler) UpdateSpeakerLabels(w http.ResponseWriter, r *http.Request) {
	userID, ok := r.Context().Value(middleware.UserIDKey).(uuid.UUID)
	if !ok {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	rawID := chi.URLParam(r, "id")
	sessionID, err := uuid.Parse(rawID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	var body model.SpeakerLabelsRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if body.SpeakerLabels == nil {
		body.SpeakerLabels = map[string]string{}
	}

	session, err := h.sessionService.GetSession(r.Context(), sessionID)
	if err != nil || session == nil || session.UserID != userID {
		writeError(w, http.StatusNotFound, "session not found")
		return
	}
	if session.Status != model.SessionStatusDone {
		writeError(w, http.StatusConflict, "session is not done")
		return
	}

	finalResult, err := h.sessionService.UpdateSpeakerLabels(r.Context(), session, body.SpeakerLabels)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"final_result": finalResult,
	})
}

// DownloadSessionArtifact godoc
// @Summary Download a session artifact
// @Description Downloads the final full audio or a chunk-level raw, normalized, or enhanced audio artifact.
// @Tags sessions
// @Produce octet-stream
// @Security BearerAuth
// @Param id path string true "Session ID"
// @Param type query string false "Artifact type: full_audio, raw, normalized, enhanced"
// @Param chunk_index query int false "Chunk index for chunk-level artifacts"
// @Success 200 {file} binary
// @Failure 400 {object} model.ErrorResponse
// @Failure 401 {object} model.ErrorResponse
// @Failure 404 {object} model.ErrorResponse
// @Router /api/v1/sessions/{id}/download [get]
func (h *SessionHandler) DownloadSessionArtifact(w http.ResponseWriter, r *http.Request) {
	userID, ok := r.Context().Value(middleware.UserIDKey).(uuid.UUID)
	if !ok {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	rawID := chi.URLParam(r, "id")
	sessionID, err := uuid.Parse(rawID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	session, err := h.sessionService.GetSession(r.Context(), sessionID)
	if err != nil || session == nil || session.UserID != userID {
		writeError(w, http.StatusNotFound, "session not found")
		return
	}

	chunks, err := h.sessionService.GetResult(r.Context(), sessionID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to get session chunks")
		return
	}

	artifactType := r.URL.Query().Get("type")
	if artifactType == "" {
		artifactType = "full_audio"
	}
	chunkIndex := 0
	if rawChunkIndex := r.URL.Query().Get("chunk_index"); rawChunkIndex != "" {
		parsedChunkIndex, err := strconv.Atoi(rawChunkIndex)
		if err != nil || parsedChunkIndex < 0 {
			writeError(w, http.StatusBadRequest, "invalid chunk_index")
			return
		}
		chunkIndex = parsedChunkIndex
	}

	key, filename, contentType, err := sessionArtifact(session, chunks, artifactType, chunkIndex)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	reader, err := h.sessionService.DownloadArtifact(r.Context(), key)
	if err != nil {
		writeError(w, http.StatusNotFound, "artifact not found")
		return
	}
	defer reader.Close()

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, filename))
	if _, err := io.Copy(w, reader); err != nil {
		return
	}
}

// StreamResults godoc
// @Summary Stream session processing events
// @Description Streams chunk and final session events through server-sent events.
// @Tags sessions
// @Produce text/event-stream
// @Security BearerAuth
// @Param id path string true "Session ID"
// @Success 200 {string} string "SSE event stream"
// @Failure 400 {object} model.ErrorResponse
// @Failure 401 {object} model.ErrorResponse
// @Failure 500 {object} model.ErrorResponse
// @Router /api/v1/sessions/{id}/stream [get]
func (h *SessionHandler) StreamResults(w http.ResponseWriter, r *http.Request) {
	rawID := chi.URLParam(r, "id")
	sessionID, err := uuid.Parse(rawID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	send := func(event model.SSEEvent) {
		data, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	lastSentIndex := 0
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-r.Context().Done():
			return

		case <-ticker.C:
			session, err := h.sessionService.GetSession(r.Context(), sessionID)
			if err != nil || session == nil {
				send(model.SSEEvent{Type: "error", Data: "session not found"})
				return
			}
			if session.Status == model.SessionStatusFailed {
				data := "session failed"
				if session.Error != nil {
					data = *session.Error
				}
				send(model.SSEEvent{Type: "error", Data: data})
				return
			}

			chunks, err := h.sessionService.GetDoneChunksSince(
				r.Context(), sessionID, lastSentIndex,
			)
			if err != nil {
				continue
			}

			for _, c := range chunks {
				var result model.ChunkResult
				if len(c.Result) > 0 {
					json.Unmarshal(c.Result, &result)
				}

				send(model.SSEEvent{
					Type:       "chunk_result",
					ChunkIndex: c.ChunkIndex,
					Data:       result,
				})

				lastSentIndex = c.ChunkIndex + 1
			}

			if session.Status == model.SessionStatusDone {
				var finalResult interface{}
				if len(session.FinalResult) > 0 {
					if err := json.Unmarshal(session.FinalResult, &finalResult); err != nil {
						finalResult = nil
					}
				}
				send(model.SSEEvent{Type: "done", Data: finalResult})
				return
			}
		}
	}
}

func sessionArtifact(session *model.Session, chunks []*model.Chunk, artifactType string, chunkIndex int) (string, string, string, error) {
	switch artifactType {
	case "full_audio":
		var result map[string]interface{}
		if len(session.FinalResult) == 0 {
			return "", "", "", fmt.Errorf("final audio is not available")
		}
		if err := json.NewDecoder(bytes.NewReader(session.FinalResult)).Decode(&result); err != nil {
			return "", "", "", fmt.Errorf("final result is invalid")
		}
		key, ok := result["audio_key"].(string)
		if !ok || key == "" {
			return "", "", "", fmt.Errorf("final audio is not available")
		}
		return key, fmt.Sprintf("session_%s_full.wav", session.ID.String()), "audio/wav", nil

	case "raw", "normalized", "enhanced":
		return chunkArtifact(session, chunks, artifactType, chunkIndex)

	default:
		return "", "", "", fmt.Errorf("unsupported artifact type")
	}
}

func chunkArtifact(session *model.Session, chunks []*model.Chunk, artifactType string, chunkIndex int) (string, string, string, error) {
	for _, chunk := range chunks {
		if chunk.ChunkIndex != chunkIndex {
			continue
		}
		switch artifactType {
		case "raw":
			return chunk.AudioKey, fmt.Sprintf("session_%s_chunk_%d_raw.webm", session.ID.String(), chunkIndex), "audio/webm", nil
		case "normalized":
			if chunk.NormalizedAudioKey == nil || *chunk.NormalizedAudioKey == "" {
				return "", "", "", fmt.Errorf("normalized audio is not available")
			}
			return *chunk.NormalizedAudioKey, fmt.Sprintf("session_%s_chunk_%d_normalized.wav", session.ID.String(), chunkIndex), "audio/wav", nil
		case "enhanced":
			if chunk.EnhancedAudioKey == nil || *chunk.EnhancedAudioKey == "" {
				return "", "", "", fmt.Errorf("enhanced audio is not available")
			}
			return *chunk.EnhancedAudioKey, fmt.Sprintf("session_%s_chunk_%d_enhanced.wav", session.ID.String(), chunkIndex), "audio/wav", nil
		}
	}
	return "", "", "", fmt.Errorf("chunk artifact is not available")
}
