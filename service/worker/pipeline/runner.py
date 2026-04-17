from pipeline import noise_reduction, diarization, asr

# порядок фиксирован
PIPELINE_ORDER = ["nr", "diar", "asr"]


def run(job: dict, audio, sr: int) -> dict:
    result = {
        "nr_applied":   False,
        "diar_applied": False,
        "asr_applied":  False,
    }

    if job.get("nr"):
        audio = noise_reduction.run(audio, sr)
        result["nr_applied"] = True

    if job.get("diar"):
        segments = diarization.run(audio, sr)
        result["diar_applied"] = True
        result["speakers"] = segments

    if job.get("asr"):
        asr_out = asr.run(audio, sr, job.get("language", "auto"))
        result["asr_applied"] = True
        result["transcript"] = asr_out["transcript"]
        result["segments"] = asr_out["segments"]

    return result