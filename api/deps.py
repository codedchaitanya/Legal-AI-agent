# api/deps.py
from fastapi import Header, HTTPException, status
from db.mongo import mongo
from db.redis_client import RedisClient

_redis: RedisClient | None = None


def get_redis() -> RedisClient:
    global _redis
    if _redis is None:
        _redis = RedisClient()
    return _redis


async def require_api_key(x_api_key: str = Header(...)) -> str:
    from core.config import settings
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return x_api_key
