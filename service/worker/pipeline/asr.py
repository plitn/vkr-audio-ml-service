from pathlib import Path
from typing import Any
import logging
import time

import config

_model = None
log = logging.getLogger(__name__)


def _get_model():
    global _model
    if _model is None:
        started = time.perf_counter()
        log.info("ASR model loading started: %s", config.PARAKEET_MODEL_NAME)
        import nemo.collections.asr as nemo_asr

        _model = nemo_asr.models.ASRModel.from_pretrained(config.PARAKEET_MODEL_NAME)
        _model.eval()
        log.info("ASR model loaded in %.2fs", time.perf_counter() - started)
    return _model


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return item.text or ""
    return str(item) if item is not None else ""


def _extract_segments(item: Any) -> list[dict]:
    timestamps = getattr(item, "timestamp", None) or getattr(item, "timestamps", None)
    if not timestamps:
        return []

    segments = timestamps.get("segment") or timestamps.get("segments") or []
    normalized = []
    for segment in segments:
        if isinstance(segment, dict):
            normalized.append(
                {
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "text": segment.get("segment") or segment.get("text") or "",
                }
            )
    return normalized


def run(wav_path: Path, language: str = "auto") -> dict:
    model = _get_model()

    started = time.perf_counter()
    log.info("ASR inference started: %s", wav_path)
    try:
        transcriptions = model.transcribe(
            [str(wav_path)],
            batch_size=1,
            timestamps=True,
        )
    except TypeError:
        transcriptions = model.transcribe([str(wav_path)], batch_size=1)
    log.info("ASR inference finished in %.2fs", time.perf_counter() - started)

    item = transcriptions[0] if transcriptions else None
    return {
        "model": config.PARAKEET_MODEL_NAME,
        "language": language,
        "transcript": _extract_text(item),
        "segments": _extract_segments(item),
    }
