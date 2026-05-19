package handler

import (
	"encoding/json"
	"fmt"
	"go-backend/internal/middleware"
	"go-backend/internal/model"
	"go-backend/internal/repository"
	"net/http"

	authsvc "go-backend/internal/service/auth"

	"github.com/google/uuid"
)

type AuthHandler struct {
	authService *authsvc.AuthService
	repo        repository.Repository
}

func NewAuthHandler(authService *authsvc.AuthService, repo repository.Repository) *AuthHandler {
	return &AuthHandler{authService: authService, repo: repo}
}

// Register godoc
// @Summary Register a user
// @Description Creates a new user account.
// @Tags auth
// @Accept json
// @Produce json
// @Param request body model.RegisterRequest true "Registration data"
// @Success 201 {object} model.User
// @Failure 400 {object} model.ErrorResponse
// @Failure 409 {object} model.ErrorResponse
// @Failure 500 {object} model.ErrorResponse
// @Router /api/v1/auth/register [post]
func (h *AuthHandler) Register(w http.ResponseWriter, r *http.Request) {
	var req model.RegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Email == "" || req.Password == "" {
		writeError(w, http.StatusBadRequest, "email and password are required")
		return
	}
	if len(req.Password) < 8 {
		writeError(w, http.StatusBadRequest, "password must be at least 8 characters")
		return
	}

	user, err := h.authService.Register(r.Context(), req)
	if err != nil {
		if err.Error() == "email already taken" {
			writeError(w, http.StatusConflict, err.Error())
			return
		}
		fmt.Println(err)
		writeError(w, http.StatusInternalServerError, "failed to register")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

// Login godoc
// @Summary Login
// @Description Authenticates a user and returns a JWT token.
// @Tags auth
// @Accept json
// @Produce json
// @Param request body model.LoginRequest true "Login data"
// @Success 200 {object} model.LoginResponse
// @Failure 400 {object} model.ErrorResponse
// @Failure 401 {object} model.ErrorResponse
// @Router /api/v1/auth/login [post]
func (h *AuthHandler) Login(w http.ResponseWriter, r *http.Request) {
	var req model.LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	resp, err := h.authService.Login(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusUnauthorized, "invalid credentials")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// Profile godoc
// @Summary Get current user profile
// @Description Returns the authenticated user profile.
// @Tags auth
// @Produce json
// @Security BearerAuth
// @Success 200 {object} model.User
// @Failure 401 {object} model.ErrorResponse
// @Failure 404 {object} model.ErrorResponse
// @Router /api/v1/auth/profile [get]
func (h *AuthHandler) Profile(w http.ResponseWriter, r *http.Request) {
	userID, ok := r.Context().Value(middleware.UserIDKey).(uuid.UUID)
	if !ok {
		writeError(w, http.StatusUnauthorized, "unauthorized")
		return
	}

	user, err := h.repo.GetUserByID(r.Context(), userID)
	if err != nil {
		writeError(w, http.StatusNotFound, "user not found")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}
