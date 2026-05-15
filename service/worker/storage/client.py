import io
from pathlib import Path

from minio import Minio

import config

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_USE_SSL,
        )
    return _client


def download_bytes(object_key: str) -> bytes:
    response = get_client().get_object(config.MINIO_BUCKET, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def download_to_path(object_key: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = download_bytes(object_key)
    path.write_bytes(data)


def upload_bytes(object_key: str, data: bytes, content_type: str = "application/octet-stream"):
    get_client().put_object(
        config.MINIO_BUCKET,
        object_key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def upload_file(object_key: str, path: Path, content_type: str = "application/octet-stream"):
    get_client().fput_object(
        config.MINIO_BUCKET,
        object_key,
        str(path),
        content_type=content_type,
    )
