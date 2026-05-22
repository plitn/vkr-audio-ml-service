# Audio ML Service

Self-hosted service for browser-based audio recording and asynchronous batch processing with automatic speech recognition, speaker diarization, and optional noise reduction.

The service records audio in the browser, uploads it in chunks, processes previous chunks while recording continues, and returns the final transcript and diarization result after the session is finalized.

## Current Status

The project is developed as part of a master's thesis on the application of machine learning methods for audio transcription, diarization, and noise reduction.

WORK DEMO:

## Prerequisites

- Docker
- Docker Compose
- Hugging Face access token if diarization is enabled

## Environment File

Create a secrets file at:

```bash
service/.env
```

Example:

```env
DB_PASSWORD=dbpass
DB_DSN=postgresql://postgres:dbpass@postgres:5432/audioservice
REDIS_URL=redis://redis:6379/0
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=secretkey
MINIO_BUCKET=audio
MINIO_USE_SSL=false
JWT_SECRET=qwerty

HF_TOKEN=hugging_face_token
```

Notes:

- `DB_PASSWORD` must match the password used inside `DB_DSN`.
- `HF_TOKEN` is required for gated Hugging Face models, especially pyannote diarization models.
- `JWT_SECRET` should be a long random value.
- `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` are used both by MinIO and by the application services.

## Run with Docker Compose

From the repository root:

```bash
cd service
docker compose up --build
```

After startup, the services are available at:

- Web application: `http://localhost:8080`
- Swagger UI: `http://localhost:8080/swagger/index.html`
- MinIO console: `http://localhost:9001`
- PostgreSQL: `localhost:5433`
- Redis: `localhost:6379`

## Local Workflow

1. Create `service/.env`;
2. Run `docker compose up --build` from the `service` directory;
3. Open `http://localhost:8080`;
4. Register a user or log in;
5. Start a recording session;
6. Stop recording and wait for backend finalization;
7. Open the result page to view the transcript, speaker blocks, and downloads.

## Project Plan

The previous README with the research and implementation plan was moved to:

```text
WORK_PLAN.md
```
