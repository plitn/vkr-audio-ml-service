import json
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

import config

_conn = None


def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(config.DB_DSN)
        _conn.autocommit = True
    return _conn


def _fetch_one(query: str, params: tuple[Any, ...]) -> dict | None:
    with get_conn().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None


def _fetch_all(query: str, params: tuple[Any, ...]) -> list[dict]:
    with get_conn().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def get_session(session_id: str) -> dict | None:
    return _fetch_one(
        """
        SELECT
            id, user_id, status, language, nr, asr, diar, diarization_mode,
            chunk_duration_sec, total_duration_sec, final_result, error,
            created_at, recording_started_at, recording_finished_at,
            processing_started_at, processing_finished_at
        FROM sessions
        WHERE id = %s
        """,
        (session_id,),
    )


def get_chunk(chunk_id: str) -> dict | None:
    return _fetch_one(
        """
        SELECT
            id, session_id, chunk_index, status, audio_key, normalized_audio_key,
            enhanced_audio_key, start_sec, duration_sec, is_final, result, error,
            created_at, processing_started_at, processing_finished_at
        FROM chunks
        WHERE id = %s
        """,
        (chunk_id,),
    )


def get_session_chunks(session_id: str) -> list[dict]:
    return _fetch_all(
        """
        SELECT
            id, session_id, chunk_index, status, audio_key, normalized_audio_key,
            enhanced_audio_key, start_sec, duration_sec, is_final, result, error,
            created_at, processing_started_at, processing_finished_at
        FROM chunks
        WHERE session_id = %s
        ORDER BY chunk_index ASC
        """,
        (session_id,),
    )


def mark_chunk_processing(chunk_id: str):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE chunks
            SET status = %s,
                processing_started_at = COALESCE(processing_started_at, now()),
                error = NULL
            WHERE id = %s AND status IN (%s, %s)
            """,
            (config.STATUS_PROCESSING, chunk_id, config.STATUS_UPLOADED, config.STATUS_QUEUED),
        )


def mark_chunk_done(
    chunk_id: str,
    normalized_audio_key: str,
    enhanced_audio_key: str | None,
    start_sec: float,
    duration_sec: float,
    result: dict,
):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE chunks
            SET status = %s,
                normalized_audio_key = %s,
                enhanced_audio_key = %s,
                start_sec = %s,
                duration_sec = %s,
                result = %s::jsonb,
                error = NULL,
                processing_finished_at = now()
            WHERE id = %s
            """,
            (
                config.STATUS_DONE,
                normalized_audio_key,
                enhanced_audio_key,
                start_sec,
                duration_sec,
                json.dumps(result),
                chunk_id,
            ),
        )


def mark_chunk_failed(chunk_id: str, error: str):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE chunks
            SET status = %s, error = %s, processing_finished_at = now()
            WHERE id = %s
            """,
            (config.STATUS_FAILED, error, chunk_id),
        )


def mark_session_processing(session_id: str):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE sessions
            SET status = %s,
                processing_started_at = COALESCE(processing_started_at, now()),
                error = NULL
            WHERE id = %s
            """,
            (config.STATUS_PROCESSING, session_id),
        )


def mark_session_done(session_id: str, total_duration_sec: float, final_result: dict):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE sessions
            SET status = %s,
                total_duration_sec = %s,
                final_result = %s::jsonb,
                error = NULL,
                processing_finished_at = now()
            WHERE id = %s
            """,
            (config.STATUS_DONE, total_duration_sec, json.dumps(final_result), session_id),
        )


def mark_session_failed(session_id: str, error: str):
    with get_conn().cursor() as cur:
        cur.execute(
            """
            UPDATE sessions
            SET status = %s, error = %s, processing_finished_at = now()
            WHERE id = %s
            """,
            (config.STATUS_FAILED, error, session_id),
        )
