package config

import (
	"context"
	"fmt"

	"github.com/joho/godotenv"
	"github.com/kelseyhightower/envconfig"
)

type Config struct {
	DB    *DB
	Redis *Redis
	Minio *MinioStorage
	JWT   *JWT
}

type DB struct {
	DSN string `envconfig:"DB_DSN"`
}

type JWT struct {
	Secret string `envconfig:"JWT_SECRET"`
}

type Redis struct {
	Url string `envconfig:"REDIS_URL"`
}

type MinioStorage struct {
	MinioEndPoint  string `envconfig:"MINIO_ENDPOINT"`
	MinioBucket    string `envconfig:"MINIO_BUCKET"`
	MinioAccessKey string `envconfig:"MINIO_ACCESS_KEY"`
	MinioSecretKey string `envconfig:"MINIO_SECRET_KEY"`
	MinioUseSSL    bool   `envconfig:"MINIO_USE_SSL"`
}

func LoadConfig(ctx context.Context) *Config {
	for _, fileName := range []string{".env.local", ".env"} {
		err := godotenv.Load(fileName)
		if err != nil {
			fmt.Printf("Error loading .env file: %s\n", err)
		}
	}
	cfg := Config{}
	if err := envconfig.Process("", &cfg); err != nil {
		fmt.Printf("cannot process envs : %s", err)
	} else {
		fmt.Printf("config initialized")
	}
	return &cfg
}
