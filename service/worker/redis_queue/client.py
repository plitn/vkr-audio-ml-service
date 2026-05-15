import json

import redis

import config

_client = None


def get_client():
    global _client
    if _client is None:
        _client = redis.from_url(config.REDIS_URL, decode_responses=True)
    return _client


def _pop(queue_name: str) -> dict:
    _, raw = get_client().brpop(queue_name)
    return json.loads(raw)


def pop_chunk() -> dict:
    return _pop(config.CHUNK_QUEUE)


def pop_session_finalize() -> dict:
    return _pop(config.SESSION_FINALIZE_QUEUE)
