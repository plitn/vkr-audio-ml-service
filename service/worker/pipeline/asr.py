import numpy as np
import nemo.collections.asr as nemo_asr

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v3"
        )
        _model.eval()
    return _model


def run(audio: np.ndarray, sr: int, language: str = "auto") -> dict:
    model = _get_model()
    transcriptions = model.transcribe([audio])

    text = transcriptions[0] if transcriptions else ""

    return {
        "transcript": text,
        "segments":   [],   # TODO: добавить word timestamps через model.transcribe с timestamps=True
    }