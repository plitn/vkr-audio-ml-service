import os

DB_DSN = os.environ["DB_DSN"]
REDIS_URL = os.environ["REDIS_URL"]

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "audio")
MINIO_USE_SSL = os.environ.get("MINIO_USE_SSL", "false").lower() == "true"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

TARGET_SR = 16000