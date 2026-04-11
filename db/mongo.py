# db/mongo.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from core.config import settings


class MongoDB:
    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self):
        self._client = AsyncIOMotorClient(settings.MONGO_URI)
        self._db = self._client.legalai
        await self._db.legal_nodes.create_index("node_id", unique=True)
        await self._db.legal_nodes.create_index("domain")
        await self._db.legal_nodes.create_index("keywords")
        await self._db.case_indexes.create_index([("case_id", 1), ("node_id", 1)])
        await self._db.conversations.create_index("session_id")

    async def disconnect(self):
        if self._client:
            self._client.close()

    def _require_connected(self):
        if self._db is None:
            raise RuntimeError("MongoDB.connect() must be awaited before accessing collections.")

    @property
    def legal_nodes(self):
        self._require_connected()
        return self._db.legal_nodes

    @property
    def case_indexes(self):
        self._require_connected()
        return self._db.case_indexes

    @property
    def conversations(self):
        self._require_connected()
        return self._db.conversations


mongo = MongoDB()
