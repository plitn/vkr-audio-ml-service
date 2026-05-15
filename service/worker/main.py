import logging
import threading
import time

import config
import db.client as db
import redis_queue.client as q
from pipeline import asr, diarization
from pipeline import runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
WORKER_VERSION = "sessions-worker-preload-and-pyannote-token-2026-05-14"


def preload_models():
    if config.PRELOAD_ASR:
        try:
            log.info("preloading ASR model")
            asr._get_model()
            log.info("ASR model preload complete")
        except Exception as exc:
            log.error("ASR model preload failed: %s", exc, exc_info=True)

    if config.PRELOAD_DIARIZATION:
        try:
            log.info("preloading diarization model")
            diarization._get_pipeline()
            log.info("diarization model preload complete")
        except Exception as exc:
            log.error("diarization model preload failed: %s", exc, exc_info=True)


def process_chunk(message: dict):
    chunk_id = message["chunk_id"]
    session_id = message["session_id"]
    log.info("processing chunk %s from session %s", chunk_id, session_id)

    chunk = db.get_chunk(chunk_id)
    if not chunk:
        raise RuntimeError(f"chunk {chunk_id} not found")

    session = db.get_session(session_id)
    if not session:
        raise RuntimeError(f"session {session_id} not found")

    db.mark_chunk_processing(chunk_id)
    output = runner.process_chunk(message, session, chunk)
    log.info("chunk %s: saving result to database", chunk_id)
    db.mark_chunk_done(
        chunk_id=chunk_id,
        normalized_audio_key=output["normalized_audio_key"],
        enhanced_audio_key=output["enhanced_audio_key"],
        start_sec=output["start_sec"],
        duration_sec=output["duration_sec"],
        result=output["result"],
    )
    log.info("chunk %s done", chunk_id)


def process_session_finalize(message: dict):
    session_id = message["session_id"]
    log.info("finalizing session %s", session_id)

    session = db.get_session(session_id)
    if not session:
        raise RuntimeError(f"session {session_id} not found")

    db.mark_session_processing(session_id)

    while True:
        chunks = db.get_session_chunks(session_id)
        if not chunks:
            time.sleep(2)
            continue

        failed = [chunk for chunk in chunks if chunk["status"] == config.STATUS_FAILED]
        if failed:
            failed_ids = ", ".join(str(chunk["id"]) for chunk in failed)
            raise RuntimeError(f"session has failed chunks: {failed_ids}")

        has_final_chunk = any(chunk["is_final"] for chunk in chunks)
        all_done = all(chunk["status"] == config.STATUS_DONE for chunk in chunks)
        if has_final_chunk and all_done:
            break

        time.sleep(2)

    session = db.get_session(session_id)
    total_duration_sec, final_result = runner.build_final_result(session, chunks)
    log.info("session %s: saving final result to database", session_id)
    db.mark_session_done(session_id, total_duration_sec, final_result)
    log.info("session %s finalized", session_id)


def chunk_loop():
    log.info("chunk worker started")
    while True:
        message = None
        try:
            message = q.pop_chunk()
            process_chunk(message)
        except Exception as exc:
            log.error("chunk processing failed: %s", exc, exc_info=True)
            if message and message.get("chunk_id"):
                try:
                    db.mark_chunk_failed(message["chunk_id"], str(exc))
                except Exception as db_exc:
                    log.error("failed to persist chunk error: %s", db_exc, exc_info=True)


def session_finalize_loop():
    log.info("session finalizer started")
    while True:
        message = None
        try:
            message = q.pop_session_finalize()
            process_session_finalize(message)
        except Exception as exc:
            log.error("session finalization failed: %s", exc, exc_info=True)
            if message and message.get("session_id"):
                try:
                    db.mark_session_failed(message["session_id"], str(exc))
                except Exception as db_exc:
                    log.error("failed to persist session error: %s", db_exc, exc_info=True)


def main():
    log.info("worker version: %s", WORKER_VERSION)
    preload_models()

    threads = [
        threading.Thread(target=chunk_loop, name="chunk-worker", daemon=True),
        threading.Thread(target=session_finalize_loop, name="session-finalizer", daemon=True),
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
