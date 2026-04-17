import json
import psycopg2
import config

_conn = None

def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(config.DB_DSN)
        _conn.autocommit = True
    return _conn


def set_status(job_id: str, status: str):
    with get_conn().cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = %s WHERE id = %s",
            (status, job_id)
        )


def save_result(job_id: str, result: dict):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status      = %s,
                result      = %s,
                finished_at = now()
            WHERE id = %s
            """,
            (config.STATUS_DONE, json.dumps(result), job_id)
        )


def set_failed(job_id: str, error: str):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status = %s, error = %s, finished_at = now()
            WHERE id = %s
            """,
            (config.STATUS_FAILED, error, job_id)
        )