import os

DB_DSN = os.environ["DB_DSN"]
REDIS_URL = os.environ["REDIS_URL"]

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "audio")
MINIO_USE_SSL = os.environ.get("MINIO_USE_SSL", "false").lower() == "true"

CHUNK_QUEUE = os.environ.get("CHUNK_QUEUE", "chunks:queue")
SESSION_FINALIZE_QUEUE = os.environ.get("SESSION_FINALIZE_QUEUE", "sessions:finalize:queue")

STATUS_UPLOADED = "uploaded"
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_FORMAT = "pcm_s16le"

PARAKEET_MODEL_NAME = os.environ.get("PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
PYANNOTE_MODEL_NAME = os.environ.get("PYANNOTE_MODEL_NAME", "pyannote/speaker-diarization-3.1")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

TMP_DIR = os.environ.get("WORKER_TMP_DIR", "/tmp/audio-worker")

PRELOAD_ASR = os.environ.get("PRELOAD_ASR", "true").lower() == "true"
PRELOAD_DIARIZATION = os.environ.get("PRELOAD_DIARIZATION", "false").lower() == "true"
