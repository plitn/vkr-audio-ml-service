package main

import (
	"context"
	"fmt"
	_ "go-backend/docs"
	"go-backend/internal/config"
	"go-backend/internal/db"
	"go-backend/internal/middleware"
	"go-backend/internal/queue"
	"go-backend/internal/repository"
	"go-backend/internal/service/auth"
	"go-backend/internal/service/handler"
	service "go-backend/internal/service/sessions"
	"go-backend/internal/storage"
	"log"
	"net/http"

	"github.com/go-chi/chi"
	"github.com/redis/go-redis/v9"
	httpSwagger "github.com/swaggo/http-swagger"
)

// @title Audio ML Service API
// @version 1.0
// @description Self-hosted browser audio recording service with asynchronous ASR, speaker diarization, and noise reduction processing.
// @BasePath /
// @schemes http
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
func main() {
	ctx := context.Background()
	cfg := config.LoadConfig(ctx)

	repoInst, err := db.New(cfg.DB)
	if err != nil {
		log.Fatalf("cannot connect to database: %v", err)
	}
	defer func() {
		err := repoInst.Close()
		if err != nil {
			log.Fatalf("cannot close repository: %v", err)
		}
	}()

	redisOpts, err := redis.ParseURL(cfg.Redis.Url)
	if err != nil {
		log.Fatalf("cannot parse redis url: %v", err)
	}
	redisClient := redis.NewClient(redisOpts)
	defer redisClient.Close()

	repo := repository.NewRepository(repoInst)
	redisQue := queue.NewRedisQueue(redisClient)

	fmt.Printf("db dsn: %s \n", cfg.DB.DSN)
	fmt.Printf("redis url: %s \n", cfg.Redis.Url)
	fmt.Printf("minio endpoint: %s \n", cfg.Minio.MinioEndPoint)
	fmt.Printf("minio access key: %s \n", cfg.Minio.MinioAccessKey)
	fmt.Printf("minio secret key: %s\n", cfg.Minio.MinioSecretKey)
	fmt.Printf("minio bucket: %s \n", cfg.Minio.MinioBucket)
	fmt.Printf("minio use ssl: %s \n", cfg.Minio.MinioUseSSL)
	jwtSecret := cfg.JWT.Secret
	if jwtSecret == "" {
		log.Fatal("JWT_SECRET is required")
	}

	storageMinio, err := storage.NewMinioStorage(cfg.Minio.MinioEndPoint, cfg.Minio.MinioAccessKey, cfg.Minio.MinioSecretKey,
		cfg.Minio.MinioBucket, cfg.Minio.MinioUseSSL)

	if err != nil {
		log.Fatalf("cannot create minio storage: %v", err)
	}
	err = storageMinio.EnsureBucket(ctx)
	if err != nil {
		log.Fatalf("cannot ensure minio bucket: %v", err)
	}

	authService := auth.NewAuthService(repo, jwtSecret)
	authHandler := handler.NewAuthHandler(authService, repo)

	sessionService := service.NewSessionService(repo, redisQue, storageMinio)
	sessionHandler := handler.NewSessionHandler(sessionService)

	authMiddleware := middleware.Auth(authService)

	mux := chi.NewRouter()
	mux.Get("/swagger/*", httpSwagger.WrapHandler)
	mux.Handle("/*", http.FileServer(http.Dir("./static")))
	mux.Get("/health", func(w http.ResponseWriter, r *http.Request) {})

	mux.Group(func(r chi.Router) {
		r.Post("/api/v1/auth/register", authHandler.Register)
		r.Post("/api/v1/auth/login", authHandler.Login)
	})

	mux.Group(func(r chi.Router) {
		r.Use(authMiddleware)
		r.Get("/api/v1/auth/profile", authHandler.Profile)

		r.Post("/api/v1/sessions", sessionHandler.CreateSession)
		r.Get("/api/v1/sessions", sessionHandler.ListSessions)
		r.Post("/api/v1/sessions/{id}/chunks", sessionHandler.AddChunk)
		r.Get("/api/v1/sessions/{id}/result", sessionHandler.GetResult)
		r.Patch("/api/v1/sessions/{id}/speaker-labels", sessionHandler.UpdateSpeakerLabels)
		r.Get("/api/v1/sessions/{id}/download", sessionHandler.DownloadSessionArtifact)
		r.Get("/api/v1/sessions/{id}/stream", sessionHandler.StreamResults)
	})

	httpServer := http.Server{
		Addr:    fmt.Sprintf("0.0.0.0:%d", 8080),
		Handler: mux,
	}
	log.Printf("Listening on port %d\n", 8080)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("cant start server: %s\n", err)
	}
}
