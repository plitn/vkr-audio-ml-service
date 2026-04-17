import json
import redis
import config


_client = None

def get_client():
    global _client
    if _client is None:
        _client = redis.from_url(config.REDIS_URL)
    return _client


def pop_job() -> dict:
    _, raw = get_client().brpop("jobs:queue")
    return json.loads(raw)