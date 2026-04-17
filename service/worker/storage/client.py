import io
import numpy as np
import soundfile as sf
from minio import Minio
import config
import librosa

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


def download_audio(audio_key: str) -> tuple[np.ndarray, int]:
    response = get_client().get_object(config.MINIO_BUCKET, audio_key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != config.TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.TARGET_SR)
        sr = config.TARGET_SR

    return audio, sr