# db/redis_client.py
import json
import hashlib
import time
import redis.asyncio as aioredis
from typing import Optional
from core.config import settings


class RedisClient:
    def __init__(self):
        self._redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,
        )

    # --- PageIndex: Client case (TTL 24h) ---
    async def set_case_pageindex(self, case_id: str, tree: dict) -> None:
        key = f"pageindex:case:{case_id}"
        await self._redis.set(key, json.dumps(tree).encode(), ex=86400)

    async def get_case_pageindex(self, case_id: str) -> Optional[dict]:
        key = f"pageindex:case:{case_id}"
        data = await self._redis.get(key)
        return json.loads(data) if data else None

    async def invalidate_case_pageindex(self, case_id: str) -> None:
        await self._redis.delete(f"pageindex:case:{case_id}")

    # --- PageIndex: Legal DB node (no TTL — static) ---
    async def set_legal_node(self, node_id: str, node: dict) -> None:
        key = f"pageindex:legal:node:{node_id}"
        await self._redis.set(key, json.dumps(node).encode())

    async def get_legal_node(self, node_id: str) -> Optional[dict]:
        key = f"pageindex:legal:node:{node_id}"
        data = await self._redis.get(key)
        return json.loads(data) if data else None

    # --- Session history (TTL 4h, max 20 messages) ---
    async def append_session_message(self, session_id: str, message: dict) -> None:
        key = f"session:{session_id}:history"
        await self._redis.rpush(key, json.dumps(message).encode())
        await self._redis.ltrim(key, -20, -1)
        await self._redis.expire(key, 14400)

    async def get_session_history(self, session_id: str) -> list:
        key = f"session:{session_id}:history"
        items = await self._redis.lrange(key, 0, -1)
        return [json.loads(item) for item in items]

    # --- Router classification cache (TTL 1h) ---
    async def set_router_result(self, doc_summary: str, domains: list) -> None:
        key = f"router:result:{hashlib.sha256(doc_summary.encode()).hexdigest()}"
        await self._redis.set(key, json.dumps(domains).encode(), ex=3600)

    async def get_router_result(self, doc_summary: str) -> Optional[list]:
        key = f"router:result:{hashlib.sha256(doc_summary.encode()).hexdigest()}"
        data = await self._redis.get(key)
        return json.loads(data) if data else None

    # --- Rate limiting (atomic sliding window per minute via SET NX + INCR pipeline) ---
    async def check_rate_limit(self, api_key: str, limit: int = 60) -> bool:
        bucket = int(time.time() // 60)
        key = f"ratelimit:{api_key}:{bucket}"
        # Atomic: SET key 1 EX 90 NX ensures TTL is always set on first write.
        # INCR on an existing key never resets TTL — no race between INCR and EXPIRE.
        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.set(key, 0, ex=90, nx=True)
            await pipe.incr(key)
            _, count = await pipe.execute()
        return count <= limit

    async def close(self):
        await self._redis.aclose()
