from pathlib import Path
import logging
import time

import config
import storage.client as storage
from pipeline import asr, audio, diarization, noise_reduction

log = logging.getLogger(__name__)


def _chunk_workdir(session_id: str, chunk_id: str) -> Path:
    path = Path(config.TMP_DIR) / session_id / "chunks" / chunk_id
    audio.ensure_tmp_dir(path)
    return path


def _session_workdir(session_id: str) -> Path:
    path = Path(config.TMP_DIR) / session_id / "final"
    audio.ensure_tmp_dir(path)
    return path


def _artifact_key(session_id: str, name: str) -> str:
    return f"sessions/{session_id}/artifacts/{name}"


def _align_segments_with_speakers(segments: list[dict], speaker_turns: list[dict]) -> list[dict]:
    aligned = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        midpoint = start + ((end - start) / 2)
        speaker = None

        for turn in speaker_turns:
            if float(turn["start"]) <= midpoint <= float(turn["end"]):
                speaker = turn["speaker"]
                break

        aligned.append(
            {
                **segment,
                "speaker": speaker or "UNKNOWN",
            }
        )
    return aligned


def process_chunk(message: dict, session: dict, chunk: dict) -> dict:
    chunk_started = time.perf_counter()
    session_id = str(session["id"])
    chunk_id = str(chunk["id"])
    chunk_index = int(chunk["chunk_index"])
    expected_speakers = session.get("expected_speakers")
    workdir = _chunk_workdir(session_id, chunk_id)

    raw_path = workdir / "raw_audio"
    normalized_path = workdir / "normalized.wav"
    enhanced_path = workdir / "enhanced.wav"

    log.info("chunk %s: downloading raw audio", chunk_id)
    storage.download_to_path(chunk["audio_key"], raw_path)
    raw_head = raw_path.read_bytes()[:16].hex()
    log.info("chunk %s: raw audio size=%d bytes head=%s", chunk_id, raw_path.stat().st_size, raw_head)

    log.info("chunk %s: decoding to PCM16 WAV", chunk_id)
    audio.decode_to_pcm16_wav(raw_path, normalized_path)

    duration = audio.duration_sec(normalized_path)
    start_sec = round(chunk_index * int(session["chunk_duration_sec"]), 3)
    normalized_key = _artifact_key(session_id, f"chunk_{chunk_index}_normalized.wav")
    log.info("chunk %s: uploading normalized audio, duration %.2fs", chunk_id, duration)
    storage.upload_file(normalized_key, normalized_path, "audio/wav")

    processing_path = normalized_path
    enhanced_key = None
    nr_applied = False
    if session["nr"]:
        log.info("chunk %s: noise reduction started", chunk_id)
        nr_started = time.perf_counter()
        noise_reduction.run(normalized_path, enhanced_path)
        log.info("chunk %s: noise reduction finished in %.2fs", chunk_id, time.perf_counter() - nr_started)
        enhanced_key = _artifact_key(session_id, f"chunk_{chunk_index}_enhanced.wav")
        storage.upload_file(enhanced_key, enhanced_path, "audio/wav")
        processing_path = enhanced_path
        nr_applied = True

    result = {
        "chunk_index": chunk_index,
        "start_sec": start_sec,
        "duration_sec": duration,
        "nr_applied": nr_applied,
        "asr_applied": False,
        "diar_applied": False,
        "diarization_mode": session["diarization_mode"],
        "expected_speakers": expected_speakers,
        "models": {
            "noise_reduction": "DeepFilterNet3" if nr_applied else None,
            "asr": config.PARAKEET_MODEL_NAME if session["asr"] else None,
            "diarization": config.PYANNOTE_MODEL_NAME if session["diar"] else None,
        },
    }

    if session["asr"]:
        log.info("chunk %s: ASR started", chunk_id)
        asr_result = asr.run(processing_path, session.get("language", "auto"))
        log.info("chunk %s: ASR produced %d chars", chunk_id, len(asr_result.get("transcript") or ""))
        result["asr_applied"] = True
        result["transcript"] = asr_result["transcript"]
        result["segments"] = asr_result["segments"]

    if session["diar"] and session["diarization_mode"] == "chunk":
        log.info("chunk %s: chunk diarization started", chunk_id)
        result["diar_applied"] = True
        result["speaker_turns"] = diarization.run(processing_path, expected_speakers)
        log.info("chunk %s: chunk diarization produced %d turns", chunk_id, len(result["speaker_turns"]))

    log.info("chunk %s: pipeline finished in %.2fs", chunk_id, time.perf_counter() - chunk_started)
    return {
        "normalized_audio_key": normalized_key,
        "enhanced_audio_key": enhanced_key,
        "start_sec": start_sec,
        "duration_sec": duration,
        "result": result,
    }


def build_final_result(session: dict, chunks: list[dict]) -> tuple[float, dict]:
    started = time.perf_counter()
    session_id = str(session["id"])
    expected_speakers = session.get("expected_speakers")
    workdir = _session_workdir(session_id)

    wav_paths = []
    transcript_parts = []
    global_segments = []
    speaker_turns = []

    for chunk in chunks:
        chunk_index = int(chunk["chunk_index"])
        result = chunk.get("result") or {}
        chunk_start = float(chunk.get("start_sec") or result.get("start_sec") or 0.0)

        audio_key = chunk.get("enhanced_audio_key") or chunk.get("normalized_audio_key")
        local_wav = workdir / f"chunk_{chunk_index}.wav"
        storage.download_to_path(audio_key, local_wav)
        wav_paths.append(local_wav)

        transcript = result.get("transcript")
        if transcript:
            transcript_parts.append(transcript)

        for segment in result.get("segments") or []:
            global_segments.append(
                {
                    **segment,
                    "start": round(float(segment.get("start", 0.0)) + chunk_start, 3),
                    "end": round(float(segment.get("end", 0.0)) + chunk_start, 3),
                    "chunk_index": chunk_index,
                }
            )

        if session["diar"] and session["diarization_mode"] == "chunk":
            for turn in result.get("speaker_turns") or []:
                speaker_turns.append(
                    {
                        **turn,
                        "start": round(float(turn["start"]) + chunk_start, 3),
                        "end": round(float(turn["end"]) + chunk_start, 3),
                        "chunk_index": chunk_index,
                    }
                )

    full_wav_path = workdir / "full.wav"
    log.info("session %s: concatenating %d chunks", session_id, len(wav_paths))
    audio.concat_wavs(wav_paths, full_wav_path)
    total_duration = audio.duration_sec(full_wav_path)
    full_audio_key = _artifact_key(session_id, "full_processing_audio.wav")
    storage.upload_file(full_audio_key, full_wav_path, "audio/wav")

    if session["diar"] and session["diarization_mode"] == "full":
        log.info("session %s: full diarization started", session_id)
        speaker_turns = diarization.run(full_wav_path, expected_speakers)
        log.info("session %s: full diarization produced %d turns", session_id, len(speaker_turns))

    aligned_segments = _align_segments_with_speakers(global_segments, speaker_turns) if speaker_turns else global_segments

    final_result = {
        "session_id": session_id,
        "language": session.get("language", "auto"),
        "diarization_mode": session["diarization_mode"],
        "expected_speakers": expected_speakers,
        "total_duration_sec": total_duration,
        "audio_key": full_audio_key,
        "transcript": " ".join(transcript_parts).strip(),
        "segments": aligned_segments,
        "speaker_turns": speaker_turns,
        "chunks": [
            {
                "chunk_index": int(chunk["chunk_index"]),
                "start_sec": float(chunk.get("start_sec") or 0.0),
                "duration_sec": float(chunk.get("duration_sec") or 0.0),
                "result": chunk.get("result") or {},
            }
            for chunk in chunks
        ],
        "models": {
            "noise_reduction": "DeepFilterNet3" if session["nr"] else None,
            "asr": config.PARAKEET_MODEL_NAME if session["asr"] else None,
            "diarization": config.PYANNOTE_MODEL_NAME if session["diar"] else None,
        },
    }
    log.info("session %s: final result built in %.2fs", session_id, time.perf_counter() - started)
    return total_duration, final_result
