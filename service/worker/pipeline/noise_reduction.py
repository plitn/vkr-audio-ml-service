from pathlib import Path

import config

_model = None
_df_state = None


def _get_model():
    global _model, _df_state
    if _model is None:
        from df.enhance import init_df

        _model, _df_state, _ = init_df()
    return _model, _df_state


def run(input_wav: Path, output_wav: Path) -> Path:
    import torch
    import torchaudio
    from df.enhance import enhance, load_audio, save_audio

    model, df_state = _get_model()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    audio, _ = load_audio(str(input_wav), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(str(output_wav), enhanced, df_state.sr())

    if df_state.sr() != config.TARGET_SR:
        waveform, sr = torchaudio.load(str(output_wav))
        waveform = torchaudio.functional.resample(waveform, sr, config.TARGET_SR)
        torchaudio.save(str(output_wav), waveform.to(torch.float32), config.TARGET_SR)

    return output_wav
