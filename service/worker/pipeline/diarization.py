from pathlib import Path
import logging
import time

import config

_pipeline = None
log = logging.getLogger(__name__)


def _patch_torchaudio_metadata():
    import torchaudio

    if hasattr(torchaudio, "AudioMetaData"):
        return

    try:
        from torchaudio.backend.common import AudioMetaData
    except ImportError:
        return

    torchaudio.AudioMetaData = AudioMetaData


def _patch_torch_load_for_pyannote():
    import torch

    if getattr(torch.load, "_vkr_weights_only_patch", False):
        return

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    patched_load._vkr_weights_only_patch = True
    torch.load = patched_load


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        started = time.perf_counter()
        log.info("diarization pipeline loading started: %s", config.PYANNOTE_MODEL_NAME)
        _patch_torchaudio_metadata()
        _patch_torch_load_for_pyannote()
        from pyannote.audio import Pipeline

        try:
            _pipeline = Pipeline.from_pretrained(
                config.PYANNOTE_MODEL_NAME,
                token=config.HF_TOKEN,
            )
        except TypeError:
            _pipeline = Pipeline.from_pretrained(
                config.PYANNOTE_MODEL_NAME,
                use_auth_token=config.HF_TOKEN,
            )
        log.info("diarization pipeline loaded in %.2fs", time.perf_counter() - started)
    return _pipeline


def run(wav_path: Path, expected_speakers: int | None = None) -> list[dict]:
    pipeline = _get_pipeline()
    started = time.perf_counter()
    kwargs = {}
    if expected_speakers:
        kwargs["num_speakers"] = int(expected_speakers)
    log.info("diarization inference started: %s, expected_speakers=%s", wav_path, expected_speakers or "auto")
    diarization = pipeline(str(wav_path), **kwargs)
    log.info("diarization inference finished in %.2fs", time.perf_counter() - started)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "speaker": speaker,
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
            }
        )
    return turns
