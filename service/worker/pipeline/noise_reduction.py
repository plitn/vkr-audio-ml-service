import numpy as np
from df.enhance import enhance, init_df

_model = None
_df_state = None


def run(audio: np.ndarray, sr: int) -> np.ndarray:
    global _model, _df_state

    if _model is None:
        _model, _df_state, _ = init_df()

    enhanced = enhance(_model, _df_state, audio)
    return enhanced