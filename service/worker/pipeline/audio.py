import subprocess
from pathlib import Path

import soundfile as sf

import config


def ensure_tmp_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def decode_to_pcm16_wav(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        str(config.TARGET_CHANNELS),
        "-ar",
        str(config.TARGET_SR),
        "-sample_fmt",
        "s16",
        "-acodec",
        config.TARGET_SAMPLE_FORMAT,
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed to decode {input_path}: {stderr}") from exc


def concat_wavs(input_paths: list[Path], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_suffix(".txt")
    list_path.write_text(
        "\n".join(f"file '{path.resolve()}'" for path in input_paths),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-ac",
        str(config.TARGET_CHANNELS),
        "-ar",
        str(config.TARGET_SR),
        "-sample_fmt",
        "s16",
        "-acodec",
        config.TARGET_SAMPLE_FORMAT,
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed to concatenate WAV files: {stderr}") from exc


def duration_sec(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return round(float(info.frames) / float(info.samplerate), 3)
