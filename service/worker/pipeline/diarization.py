import numpy as np


def run(audio: np.ndarray, sr: int) -> list[dict]:
    """
    stub
    """
    duration = len(audio) / sr
    return [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": round(duration, 3),
        }
    ]