import logging
import config
import db.client as db
import queue.client as q
import storage.client as storage
from pipeline import runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


def process(job: dict):
    job_id = job["job_id"]
    log.info(f"processing job {job_id}")

    db.set_status(job_id, config.STATUS_RUNNING)

    audio, sr = storage.download_audio(job["audio_key"])
    result = runner.run(job, audio, sr)

    db.save_result(job_id, result)
    log.info(f"job {job_id} done")


def main():
    log.info("worker started")
    while True:
        try:
            job = q.pop_job()
            process(job)
        except Exception as e:
            job_id = job.get("job_id") if "job" in dir() else "unknown"
            log.error(f"job {job_id} failed: {e}", exc_info=True)
            try:
                db.set_failed(job_id, str(e))
            except Exception as db_err:
                log.error(f"failed to write error to db: {db_err}")


if __name__ == "__main__":
    main()