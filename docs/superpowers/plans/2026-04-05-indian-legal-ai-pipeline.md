# Indian Legal AI Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid Indian legal AI assistant that does case research, document Q&A, and argument generation using PageIndex-based retrieval, domain-specific QLoRA adapters, and a CoT agentic pipeline with critic scoring.

**Architecture:** FastAPI backend + Gradio frontend. Two PageIndex instances (static legal DB + per-case client docs). Domain router selects QLoRA adapters (PEFT). Big model API handles argument generation with agentic tool calls. Critic model scores and filters arguments.

**Tech Stack:** Python 3.11, FastAPI, Gradio, PEFT + TRL + Transformers, MongoDB, Redis, PostgreSQL, MinIO, Docker Compose, Qwen2.5-7B-Instruct base model, GPT-4.1 API.

---

## Engineering Manager Decision Log

All architectural decisions made upfront. Do not relitigate these during implementation.

### DB Architecture Decision

| Store | Role | Why |
|-------|------|-----|
| **MongoDB** | Primary document store — PageIndex trees, case documents, legal DB nodes | Hierarchical JSON maps directly to PageIndex tree structure. Rich querying on nested nodes. Flexible schema handles different doc types. |
| **Redis** | Primary cache + session store | Sub-millisecond reads for hot case data. PageIndex trees hit on every query — must be in-memory. Session history needs instant access. |
| **PostgreSQL** | Relational metadata — cases, users, audit logs, benchmark results | ACID guarantees for case management. Foreign keys between cases/lawyers/documents. Audit trail is non-negotiable for legal systems. |
| **MinIO** | Object storage — raw PDFs, adapter weights, exported indexes | Files don't live in databases. Self-hosted S3-compatible — upgrade to AWS S3 later without code changes. |
| **No vector DB** | Intentional — vectorless by design | PageIndex + reasoning-based retrieval replaces vector search entirely. Adding a vector DB would contradict the core architecture. |

### Redis Cache Key Design

```
pageindex:case:{case_id}              → full client PageIndex JSON     TTL: 24h
pageindex:legal:node:{node_id}        → single legal DB node           TTL: none (static)
pageindex:legal:subtree:{node_id}     → subtree under node             TTL: none (static)
session:{session_id}:history          → last 20 messages (JSON array)  TTL: 4h
router:result:{sha256(doc_summary)}   → domain classification labels   TTL: 1h
adapter:selection:{sha256(domains)}   → selected adapter names         TTL: 1h
ratelimit:{api_key}:{minute_bucket}   → request count                  TTL: 90s
stream:{session_id}:argument          → pub/sub channel for streaming  TTL: 10min
```

**Cache invalidation rules:**
- `pageindex:case:{case_id}` → DELETE on any new document upload for that case
- `session:*:history` → APPEND on each message, never full delete until TTL
- Legal DB cache → never expire (static). Only purge on deliberate `scripts/invalidate_legal_cache.py` run after law amendment.

### LoRA Rank Decisions (locked in)

| Adapter | Rank (r) | Alpha | Dropout | Target modules |
|---------|----------|-------|---------|----------------|
| Domain Q&A adapters (×11) | 32 | 64 | 0.05 | q,k,v,o projections |
| Router/classifier | 16 | 32 | 0.05 | q,v projections |
| Argument generation | 128 | 256 | 0.05 | q,k,v,o + gate,up,down (MLP) |
| Critic/reward model | 64 | 128 | 0.05 | q,k,v,o projections |

### API Key / Model Routing Decision

- Case research + Doc Q&A: small model + LoRA adapter (free, fast, local)
- Argument generation Stage 2: GPT-4.1 API (best reasoning, worth the cost)
- Critic Phase 1: GPT-4.1 API as judge (~$0.02/eval)
- Critic Phase 2 (later): fine-tuned reward model (replaces API call)

### Hallucination Guard Decision

Every response goes through a post-generation citation validator before being returned to the user. Any cited section (e.g., "BNS Section 103") is verified against the PageIndex response context. If a citation appears that was not in retrieved context → flag with `[UNVERIFIED]` tag, do not silently pass.

---

## Repository Structure

```
legal-ai/
├── api/
│   ├── main.py                         # FastAPI app, lifespan, CORS
│   ├── routes/
│   │   ├── cases.py                    # POST /cases, GET /cases/{id}
│   │   ├── documents.py                # POST /cases/{id}/documents
│   │   ├── query.py                    # POST /cases/{id}/query (research + Q&A)
│   │   └── arguments.py                # POST /cases/{id}/arguments
│   ├── middleware/
│   │   ├── auth.py                     # API key validation
│   │   └── rate_limit.py               # Redis sliding window rate limiter
│   └── deps.py                         # FastAPI dependency injection (db, redis, model)
├── core/
│   ├── ingestion/
│   │   ├── pdf_loader.py               # PDF → list of page images + raw text
│   │   ├── summarizer.py               # LLM-based extractive summary + NER
│   │   └── doc_classifier.py           # Classify doc as FIR/contract/affidavit/etc
│   ├── indexing/
│   │   ├── pageindex_builder.py        # Build PageIndex tree from doc pages
│   │   ├── pageindex_query.py          # Navigate/fetch nodes from tree
│   │   └── legal_db_precompute.py      # One-time: build legal DB index
│   ├── routing/
│   │   ├── domain_router.py            # Multi-label domain classifier (zero-shot → BERT)
│   │   └── adapter_selector.py         # Map domain labels → adapter names
│   ├── reasoning/
│   │   ├── lora_engine.py              # Load base model, manage adapter switching
│   │   ├── context_merger.py           # Merge client PageIndex + legal DB context
│   │   ├── case_research.py            # Case research query pipeline
│   │   ├── doc_qa.py                   # Document Q&A pipeline
│   │   └── argument_gen.py             # CoT + agentic argument generation
│   ├── tools/                          # Agentic tool definitions (called mid-CoT)
│   │   ├── fetch_legal_db.py           # tool: fetch_legal_db(node_id)
│   │   ├── fetch_case_docs.py          # tool: fetch_case_docs(doc_id, page)
│   │   ├── resolve_cross_ref.py        # tool: resolve_cross_ref(citation_text)
│   │   └── search_precedents.py        # tool: search_precedents(query)
│   ├── critic/
│   │   ├── scorer.py                   # Score argument on 4 dimensions
│   │   └── filter.py                   # Top-K filter + threshold + retry logic
│   └── validation/
│       └── citation_validator.py       # Post-generation hallucination guard
├── db/
│   ├── mongo.py                        # MongoDB client, collection accessors
│   ├── redis_client.py                 # Redis client, get/set/delete helpers
│   ├── postgres.py                     # SQLAlchemy models + async engine
│   └── storage.py                      # MinIO client, upload/download
├── training/
│   ├── data_prep/
│   │   ├── kanoon_scraper.py           # Scrape Indian Kanoon judgments
│   │   ├── format_qa_pairs.py          # Format to instruction-tuning JSONL
│   │   └── domain_splitter.py          # Split dataset by legal domain
│   ├── train_adapter.py                # QLoRA training script (PEFT + TRL)
│   ├── eval_adapter.py                 # InLegalBench evaluation runner
│   └── configs/
│       ├── base.yaml                   # Shared training config
│       ├── criminal.yaml               # Criminal domain overrides
│       └── argument_gen.yaml           # Argument gen adapter config (r=128)
├── ui/
│   └── app.py                          # Gradio chat interface
├── tests/
│   ├── unit/
│   │   ├── test_pdf_loader.py
│   │   ├── test_summarizer.py
│   │   ├── test_pageindex_builder.py
│   │   ├── test_pageindex_query.py
│   │   ├── test_domain_router.py
│   │   ├── test_citation_validator.py
│   │   └── test_critic_scorer.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_case_research_pipeline.py
│   │   └── test_argument_gen_pipeline.py
│   └── eval/
│       ├── run_inlegalbench.py
│       └── run_domain_eval.py
├── scripts/
│   ├── precompute_legal_db.py          # Run once: build + store legal DB PageIndex
│   ├── seed_iit_patna_adapter.py       # Load IIT Patna adapter as criminal_v0
│   └── invalidate_legal_cache.py       # Run after law amendments
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.training
│   └── docker-compose.yml
├── .env.example
├── requirements.txt
└── pytest.ini
```

---

## Phase 1 — Foundation (Weeks 1–4)

**Deliverable:** Working system that ingests client documents, builds PageIndex, does basic Q&A using IIT Patna adapter as `criminal_v0`, and serves via FastAPI + Gradio. Fully benchmarked on InLegalBench zero-shot baseline.

---

### Task 1: Project Scaffold + Docker Infrastructure

**Files:**
- Create: `docker/docker-compose.yml`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `pytest.ini`

- [ ] **Step 1: Write docker-compose.yml**

```yaml
# docker/docker-compose.yml
version: "3.9"

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    env_file: ../.env
    depends_on:
      - mongo
      - redis
      - postgres
      - minio
    volumes:
      - ../adapters:/app/adapters

  mongo:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: legalai

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: legalai
      POSTGRES_USER: legalai
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ACCESS_KEY}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY}
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  gradio:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    command: python ui/app.py
    ports:
      - "7860:7860"
    env_file: ../.env
    depends_on:
      - api

volumes:
  mongo_data:
  redis_data:
  postgres_data:
  minio_data:
```

- [ ] **Step 2: Write .env.example**

```bash
# .env.example
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

MONGO_URI=mongodb://mongo:27017/legalai
REDIS_URL=redis://redis:6379/0
POSTGRES_URL=postgresql+asyncpg://legalai:${POSTGRES_PASSWORD}@postgres:5432/legalai
POSTGRES_PASSWORD=changeme

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=changeme
MINIO_BUCKET_DOCS=legal-docs
MINIO_BUCKET_ADAPTERS=lora-adapters

BASE_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
ADAPTERS_DIR=/app/adapters

# Thresholds
CRITIC_THRESHOLD=0.72
ARGUMENT_K=5
SECONDARY_ADAPTER_CONFIDENCE=0.40
```

- [ ] **Step 3: Write requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
gradio==4.44.0
pydantic==2.9.0
pydantic-settings==2.5.0

# ML
torch==2.4.0
transformers==4.46.1
peft==0.13.2
trl==0.11.0
accelerate==0.34.0
bitsandbytes==0.44.0
sentencepiece==0.2.0

# PDF processing
pypdf2==3.0.1
pdf2image==1.17.0
Pillow==10.4.0

# Databases
motor==3.5.0              # async MongoDB
redis[hiredis]==5.1.0
sqlalchemy[asyncio]==2.0.35
asyncpg==0.29.0
alembic==1.13.3
minio==7.2.9

# Utilities
httpx==0.27.2
python-multipart==0.0.12
python-dotenv==1.0.1
tenacity==9.0.0
structlog==24.4.0

# Testing
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-mock==3.14.0
httpx==0.27.2
```

- [ ] **Step 4: Write pytest.ini**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 5: Write Dockerfile.api**

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Start services and verify**

```bash
cp .env.example .env
# Fill in your API keys in .env
docker compose -f docker/docker-compose.yml up -d mongo redis postgres minio
docker compose -f docker/docker-compose.yml ps
```

Expected: all 4 services show `Up` status.

- [ ] **Step 7: Commit**

```bash
git add docker/ .env.example requirements.txt pytest.ini
git commit -m "feat: project scaffold with docker-compose, postgres, mongo, redis, minio"
```

---

### Task 2: Database Clients + PostgreSQL Schema

**Files:**
- Create: `db/mongo.py`
- Create: `db/redis_client.py`
- Create: `db/postgres.py`
- Create: `db/storage.py`
- Create: `tests/unit/test_db_clients.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_db_clients.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_redis_set_and_get_pageindex():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.set = AsyncMock(return_value=True)
    client._redis.get = AsyncMock(return_value=b'{"node_id": "BNS-103"}')

    await client.set_case_pageindex("CASE-001", {"node_id": "BNS-103"})
    result = await client.get_case_pageindex("CASE-001")

    assert result == {"node_id": "BNS-103"}
    client._redis.set.assert_called_once()

@pytest.mark.asyncio
async def test_redis_delete_case_pageindex():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.delete = AsyncMock(return_value=1)

    await client.invalidate_case_pageindex("CASE-001")
    client._redis.delete.assert_called_with("pageindex:case:CASE-001")

@pytest.mark.asyncio
async def test_redis_append_session_history():
    from db.redis_client import RedisClient
    client = RedisClient.__new__(RedisClient)
    client._redis = AsyncMock()
    client._redis.rpush = AsyncMock(return_value=1)
    client._redis.ltrim = AsyncMock(return_value=True)
    client._redis.expire = AsyncMock(return_value=True)

    msg = {"role": "user", "content": "What is Section 103 BNS?"}
    await client.append_session_message("SESSION-1", msg)
    client._redis.rpush.assert_called_once()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/test_db_clients.py -v
```

Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Write db/redis_client.py**

```python
# db/redis_client.py
import json
import hashlib
import redis.asyncio as aioredis
from typing import Any, Optional
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
        await self._redis.ltrim(key, -20, -1)  # keep last 20 only
        await self._redis.expire(key, 14400)   # 4h TTL

    async def get_session_history(self, session_id: str) -> list[dict]:
        key = f"session:{session_id}:history"
        items = await self._redis.lrange(key, 0, -1)
        return [json.loads(item) for item in items]

    # --- Router classification cache (TTL 1h) ---
    async def set_router_result(self, doc_summary: str, domains: list[str]) -> None:
        key = f"router:result:{hashlib.sha256(doc_summary.encode()).hexdigest()}"
        await self._redis.set(key, json.dumps(domains).encode(), ex=3600)

    async def get_router_result(self, doc_summary: str) -> Optional[list[str]]:
        key = f"router:result:{hashlib.sha256(doc_summary.encode()).hexdigest()}"
        data = await self._redis.get(key)
        return json.loads(data) if data else None

    # --- Rate limiting (sliding window per minute) ---
    async def check_rate_limit(self, api_key: str, limit: int = 60) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        import time
        bucket = int(time.time() // 60)
        key = f"ratelimit:{api_key}:{bucket}"
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, 90)
        return count <= limit

    async def close(self):
        await self._redis.aclose()
```

- [ ] **Step 4: Write db/mongo.py**

```python
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
        # Create indexes on startup
        await self._db.legal_nodes.create_index("node_id", unique=True)
        await self._db.legal_nodes.create_index("domain")
        await self._db.legal_nodes.create_index("keywords")
        await self._db.case_indexes.create_index([("case_id", 1), ("node_id", 1)])
        await self._db.conversations.create_index("session_id")

    async def disconnect(self):
        if self._client:
            self._client.close()

    @property
    def legal_nodes(self):
        return self._db.legal_nodes

    @property
    def case_indexes(self):
        return self._db.case_indexes

    @property
    def conversations(self):
        return self._db.conversations


mongo = MongoDB()
```

- [ ] **Step 5: Write db/postgres.py**

```python
# db/postgres.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, ForeignKey, Text, Float, JSON
from datetime import datetime, UTC
from typing import Optional
from core.config import settings


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    lawyer_id: Mapped[str] = mapped_column(String(100))
    domains: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    documents: Mapped[list["Document"]] = relationship(back_populates="case")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"))
    doc_type: Mapped[str] = mapped_column(String(50))  # FIR/contract/affidavit
    filename: Mapped[str] = mapped_column(String(255))
    page_count: Mapped[int] = mapped_column(default=0)
    storage_path: Mapped[str] = mapped_column(String(500))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    case: Mapped["Case"] = relationship(back_populates="documents")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(50))
    session_id: Mapped[str] = mapped_column(String(100))
    query_type: Mapped[str] = mapped_column(String(30))  # research/qa/argument
    query: Mapped[str] = mapped_column(Text)
    adapters_used: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    api_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


engine = create_async_engine(settings.POSTGRES_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

- [ ] **Step 6: Write core/config.py (needed by db modules)**

```python
# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    MONGO_URI: str = "mongodb://localhost:27017/legalai"
    REDIS_URL: str = "redis://localhost:6379/0"
    POSTGRES_URL: str = "postgresql+asyncpg://legalai:changeme@localhost:5432/legalai"
    POSTGRES_PASSWORD: str = "changeme"

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "changeme"
    MINIO_BUCKET_DOCS: str = "legal-docs"
    MINIO_BUCKET_ADAPTERS: str = "lora-adapters"

    BASE_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    ADAPTERS_DIR: str = "./adapters"

    CRITIC_THRESHOLD: float = 0.72
    ARGUMENT_K: int = 5
    SECONDARY_ADAPTER_CONFIDENCE: float = 0.40


settings = Settings()
```

- [ ] **Step 7: Run tests — expect PASS**

```bash
pytest tests/unit/test_db_clients.py -v
```

Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add db/ core/config.py tests/unit/test_db_clients.py
git commit -m "feat: database clients — mongo, redis (with cache key design), postgres schema"
```

---

### Task 3: PDF Ingestion + Extractive Summarizer

**Files:**
- Create: `core/ingestion/pdf_loader.py`
- Create: `core/ingestion/summarizer.py`
- Create: `core/ingestion/doc_classifier.py`
- Create: `tests/unit/test_pdf_loader.py`
- Create: `tests/unit/test_summarizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_pdf_loader.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_pdf_to_pages_returns_list_of_page_dicts():
    from core.ingestion.pdf_loader import load_pdf_pages
    with patch("core.ingestion.pdf_loader.convert_from_path") as mock_convert:
        mock_img = MagicMock()
        mock_convert.return_value = [mock_img, mock_img]
        pages = load_pdf_pages("fake.pdf")
        assert len(pages) == 2
        assert "page_number" in pages[0]
        assert "image" in pages[0]


def test_pdf_page_numbers_are_one_indexed():
    from core.ingestion.pdf_loader import load_pdf_pages
    with patch("core.ingestion.pdf_loader.convert_from_path") as mock_convert:
        mock_img = MagicMock()
        mock_convert.return_value = [mock_img]
        pages = load_pdf_pages("fake.pdf")
        assert pages[0]["page_number"] == 1
```

```python
# tests/unit/test_summarizer.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_summarizer_returns_required_fields():
    from core.ingestion.summarizer import summarize_document
    mock_response = {
        "summary": "FIR filed against accused for theft",
        "entities": {"persons": ["Raju"], "dates": ["2024-01-15"]},
        "sections_mentioned": [{"raw": "Section 379 IPC", "bns_equivalent": "BNS 303"}],
        "doc_type": "FIR",
        "keywords": ["theft", "FIR", "accused"]
    }
    with patch("core.ingestion.summarizer.call_llm", new=AsyncMock(return_value=mock_response)):
        result = await summarize_document("Some FIR text here")
        assert "summary" in result
        assert "sections_mentioned" in result
        assert "doc_type" in result
        assert "keywords" in result
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/test_pdf_loader.py tests/unit/test_summarizer.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write core/ingestion/pdf_loader.py**

```python
# core/ingestion/pdf_loader.py
from pdf2image import convert_from_path
from pathlib import Path
from typing import BinaryIO
import tempfile
import os


def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Convert PDF to list of page dicts.
    Each dict: {page_number: int (1-indexed), image: PIL.Image}
    """
    images = convert_from_path(str(pdf_path), dpi=150)
    return [
        {"page_number": i + 1, "image": img}
        for i, img in enumerate(images)
    ]


def load_pdf_from_bytes(data: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        return load_pdf_pages(tmp_path)
    finally:
        os.unlink(tmp_path)
```

- [ ] **Step 4: Write core/ingestion/summarizer.py**

```python
# core/ingestion/summarizer.py
import json
from openai import AsyncOpenAI
from core.config import settings

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# IPC → BNS section mapping (extend this as needed)
IPC_TO_BNS = {
    "302": "BNS 103", "304": "BNS 105", "307": "BNS 109",
    "376": "BNS 63",  "420": "BNS 318", "379": "BNS 303",
    "498A": "BNS 85", "406": "BNS 316",
}

SUMMARIZE_PROMPT = """You are an Indian legal document analyst.
Analyse the following document and return ONLY a valid JSON object with these exact keys:
{
  "summary": "2-3 sentence extractive summary using sentences from the document",
  "entities": {"persons": [...], "dates": [...], "locations": [...], "amounts": [...]},
  "sections_mentioned": [{"raw": "Section 302 IPC", "bns_equivalent": "BNS 103"}],
  "doc_type": "FIR|contract|affidavit|judgment|chargesheet|notice|other",
  "keywords": ["max 10 most legally relevant keywords"]
}
Document:
"""


async def call_llm(text: str) -> dict:
    response = await _client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": SUMMARIZE_PROMPT + text[:8000]}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


async def summarize_document(text: str) -> dict:
    result = await call_llm(text)
    # Enrich sections_mentioned with BNS mapping
    for sec in result.get("sections_mentioned", []):
        raw = sec.get("raw", "")
        for ipc_num, bns_equiv in IPC_TO_BNS.items():
            if ipc_num in raw and not sec.get("bns_equivalent"):
                sec["bns_equivalent"] = bns_equiv
    return result
```

- [ ] **Step 5: Write core/ingestion/doc_classifier.py**

```python
# core/ingestion/doc_classifier.py
# Thin wrapper — doc_type is already returned by summarizer.
# This module exists as an extension point if we add a dedicated
# classifier model later (Phase 2 BERT-based option).

VALID_DOC_TYPES = frozenset([
    "FIR", "contract", "affidavit", "judgment",
    "chargesheet", "notice", "other"
])


def normalise_doc_type(raw_type: str) -> str:
    """Normalise LLM-returned doc_type to a known enum value."""
    cleaned = raw_type.strip().lower()
    mapping = {
        "fir": "FIR", "first information report": "FIR",
        "contract": "contract", "agreement": "contract",
        "affidavit": "affidavit", "sworn statement": "affidavit",
        "judgment": "judgment", "order": "judgment", "verdict": "judgment",
        "chargesheet": "chargesheet", "charge sheet": "chargesheet",
        "notice": "notice", "legal notice": "notice",
    }
    return mapping.get(cleaned, "other")
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
pytest tests/unit/test_pdf_loader.py tests/unit/test_summarizer.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add core/ingestion/ tests/unit/test_pdf_loader.py tests/unit/test_summarizer.py
git commit -m "feat: pdf ingestion, extractive summarizer with IPC→BNS mapping, doc classifier"
```

---

### Task 4: PageIndex Builder — Client Documents

**Files:**
- Create: `core/indexing/pageindex_builder.py`
- Create: `core/indexing/pageindex_query.py`
- Create: `tests/unit/test_pageindex_builder.py`
- Create: `tests/unit/test_pageindex_query.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_pageindex_builder.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_build_pageindex_returns_tree_with_required_fields():
    from core.indexing.pageindex_builder import build_case_pageindex
    pages = [
        {"page_number": 1, "image": None, "text": "FIR filed on 2024-01-15. Accused Raju arrested."},
        {"page_number": 2, "image": None, "text": "Complainant states theft occurred at midnight."},
    ]
    with patch("core.indexing.pageindex_builder.generate_page_summary",
               new=AsyncMock(return_value="FIR details on page 1")):
        tree = await build_case_pageindex("CASE-001", "doc-1", "FIR", pages)
    assert tree["case_id"] == "CASE-001"
    assert "sub_nodes" in tree
    assert len(tree["sub_nodes"]) == 2
    assert tree["sub_nodes"][0]["page_number"] == 1
    assert "summary" in tree["sub_nodes"][0]
    assert "node_id" in tree["sub_nodes"][0]


@pytest.mark.asyncio
async def test_pageindex_node_id_format():
    from core.indexing.pageindex_builder import build_case_pageindex
    with patch("core.indexing.pageindex_builder.generate_page_summary",
               new=AsyncMock(return_value="summary")):
        tree = await build_case_pageindex("CASE-001", "DOC-1", "FIR",
                                          [{"page_number": 1, "image": None, "text": "test"}])
    node_id = tree["sub_nodes"][0]["node_id"]
    assert node_id == "CASE-001-DOC-1-P1"
```

```python
# tests/unit/test_pageindex_query.py
import pytest


def test_fetch_node_by_id_returns_correct_node():
    from core.indexing.pageindex_query import fetch_node_by_id
    tree = {
        "node_id": "CASE-001",
        "sub_nodes": [
            {"node_id": "CASE-001-DOC-1-P1", "summary": "page 1", "sub_nodes": []},
            {"node_id": "CASE-001-DOC-1-P2", "summary": "page 2", "sub_nodes": []},
        ]
    }
    result = fetch_node_by_id(tree, "CASE-001-DOC-1-P2")
    assert result["summary"] == "page 2"


def test_fetch_node_by_id_returns_none_for_missing():
    from core.indexing.pageindex_query import fetch_node_by_id
    tree = {"node_id": "CASE-001", "sub_nodes": []}
    result = fetch_node_by_id(tree, "NONEXISTENT")
    assert result is None
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/unit/test_pageindex_builder.py tests/unit/test_pageindex_query.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write core/indexing/pageindex_builder.py**

```python
# core/indexing/pageindex_builder.py
import json
from openai import AsyncOpenAI
from core.config import settings

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

PAGE_SUMMARY_PROMPT = """Summarise this legal document page in 1-2 sentences.
Focus on: what legal event/content is described, any section numbers mentioned, key parties.
Return ONLY the summary sentence(s), nothing else.

Page text:
"""


async def generate_page_summary(page_text: str) -> str:
    response = await _client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": PAGE_SUMMARY_PROMPT + page_text[:3000]}],
        temperature=0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


async def build_case_pageindex(
    case_id: str,
    doc_id: str,
    doc_type: str,
    pages: list[dict],
) -> dict:
    """
    Build a PageIndex tree for a single document.
    pages: list of {page_number: int, image: PIL.Image|None, text: str}
    Returns a tree dict ready to store in MongoDB / cache in Redis.
    """
    sub_nodes = []
    for page in pages:
        page_num = page["page_number"]
        text = page.get("text", "")
        summary = await generate_page_summary(text) if text.strip() else f"{doc_type} page {page_num}"
        sub_nodes.append({
            "node_id": f"{case_id}-{doc_id}-P{page_num}",
            "title": f"{doc_type} — Page {page_num}",
            "page_number": page_num,
            "summary": summary,
            "doc_id": doc_id,
            "sub_nodes": [],
        })

    return {
        "node_id": f"{case_id}-{doc_id}",
        "case_id": case_id,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": f"{doc_type} ({doc_id})",
        "total_pages": len(pages),
        "sub_nodes": sub_nodes,
    }
```

- [ ] **Step 4: Write core/indexing/pageindex_query.py**

```python
# core/indexing/pageindex_query.py
from typing import Optional


def fetch_node_by_id(tree: dict, node_id: str) -> Optional[dict]:
    """Recursively search tree for a node by node_id."""
    if tree.get("node_id") == node_id:
        return tree
    for child in tree.get("sub_nodes", []):
        result = fetch_node_by_id(child, node_id)
        if result:
            return result
    return None


def get_toc_summary(tree: dict, max_depth: int = 2) -> str:
    """
    Return a flat text Table of Contents for LLM consumption.
    Format: node_id | title | summary
    """
    lines = []

    def _walk(node: dict, depth: int):
        if depth > max_depth:
            return
        indent = "  " * depth
        lines.append(f"{indent}[{node['node_id']}] {node.get('title', '')} — {node.get('summary', '')}")
        for child in node.get("sub_nodes", []):
            _walk(child, depth + 1)

    _walk(tree, 0)
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/unit/test_pageindex_builder.py tests/unit/test_pageindex_query.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add core/indexing/ tests/unit/test_pageindex_builder.py tests/unit/test_pageindex_query.py
git commit -m "feat: PageIndex builder and tree query for client documents"
```

---

### Task 5: Legal DB PageIndex Pre-computation Script

**Files:**
- Create: `core/indexing/legal_db_precompute.py`
- Create: `scripts/precompute_legal_db.py`
- Create: `data/legal_db/bns_sections.json` (sample — 5 sections for testing)

- [ ] **Step 1: Create sample BNS test data**

```json
// data/legal_db/bns_sections.json
[
  {
    "act": "Bharatiya Nyaya Sanhita 2023",
    "section_number": "103",
    "title": "Murder",
    "old_equivalent": "IPC Section 302",
    "text": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
    "domain": "criminal",
    "keywords": ["murder", "culpable homicide", "death penalty", "imprisonment for life"],
    "page_range": [88, 89]
  },
  {
    "act": "Bharatiya Nyaya Sanhita 2023",
    "section_number": "318",
    "title": "Cheating",
    "old_equivalent": "IPC Section 420",
    "text": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property...",
    "domain": "criminal",
    "keywords": ["cheating", "fraud", "deception", "property"],
    "page_range": [201, 202]
  }
]
```

- [ ] **Step 2: Write core/indexing/legal_db_precompute.py**

```python
# core/indexing/legal_db_precompute.py
import json
from pathlib import Path
from openai import AsyncOpenAI
from core.config import settings

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

LEGAL_SUMMARY_PROMPT = """You are an Indian legal expert.
Summarise this legal section in 2-3 sentences for a lawyer.
Include: what the section covers, what it prohibits/permits, what punishment/remedy it provides.
Return ONLY the summary.

Section text:
"""


async def generate_section_summary(section_text: str) -> str:
    response = await _client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": LEGAL_SUMMARY_PROMPT + section_text[:2000]}],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


async def build_legal_node(section: dict) -> dict:
    """Convert a raw section dict to a PageIndex node with LLM summary."""
    summary = await generate_section_summary(section["text"])
    act_code = section["act"].replace(" ", "_").replace(",", "")[:20]
    node_id = f"{act_code}-{section['section_number']}"
    return {
        "node_id": node_id,
        "title": f"Section {section['section_number']} — {section['title']}",
        "act": section["act"],
        "section_number": section["section_number"],
        "old_equivalent": section.get("old_equivalent", ""),
        "summary": summary,
        "text": section["text"],
        "keywords": section.get("keywords", []),
        "domain": section["domain"],
        "page_range": section.get("page_range", []),
        "sub_nodes": [],
    }


async def precompute_from_file(json_path: str) -> list[dict]:
    """Load sections from JSON, generate summaries, return list of nodes."""
    sections = json.loads(Path(json_path).read_text())
    nodes = []
    for section in sections:
        node = await build_legal_node(section)
        nodes.append(node)
        print(f"  Built node: {node['node_id']}")
    return nodes
```

- [ ] **Step 3: Write scripts/precompute_legal_db.py**

```python
#!/usr/bin/env python3
# scripts/precompute_legal_db.py
"""
Run once to build and persist the Legal DB PageIndex.
Usage: python scripts/precompute_legal_db.py --input data/legal_db/bns_sections.json
"""
import asyncio
import argparse
from core.indexing.legal_db_precompute import precompute_from_file
from db.mongo import mongo
from db.redis_client import RedisClient


async def main(input_path: str):
    await mongo.connect()
    redis = RedisClient()

    print(f"Building legal DB index from {input_path}...")
    nodes = await precompute_from_file(input_path)

    print(f"Storing {len(nodes)} nodes in MongoDB...")
    for node in nodes:
        await mongo.legal_nodes.replace_one(
            {"node_id": node["node_id"]},
            node,
            upsert=True,
        )

    print("Warming Redis cache for legal nodes...")
    for node in nodes:
        await redis.set_legal_node(node["node_id"], node)

    print(f"Done. {len(nodes)} nodes indexed.")
    await mongo.disconnect()
    await redis.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.input))
```

- [ ] **Step 4: Run the precompute script against sample data**

```bash
python scripts/precompute_legal_db.py --input data/legal_db/bns_sections.json
```

Expected output:
```
Building legal DB index from data/legal_db/bns_sections.json...
  Built node: Bharatiya_Nyaya_Sanhita-103
  Built node: Bharatiya_Nyaya_Sanhita-318
Storing 2 nodes in MongoDB...
Warming Redis cache for legal nodes...
Done. 2 nodes indexed.
```

- [ ] **Step 5: Commit**

```bash
git add core/indexing/legal_db_precompute.py scripts/precompute_legal_db.py data/
git commit -m "feat: legal DB PageIndex pre-computation script with Redis warming"
```

---

### Task 6: Domain Router (Zero-Shot Phase 1)

**Files:**
- Create: `core/routing/domain_router.py`
- Create: `core/routing/adapter_selector.py`
- Create: `tests/unit/test_domain_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_domain_router.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_router_returns_domain_list():
    from core.routing.domain_router import classify_domains
    mock_result = [{"domain": "criminal", "confidence": 0.92},
                   {"domain": "constitutional", "confidence": 0.45}]
    with patch("core.routing.domain_router.call_classifier",
               new=AsyncMock(return_value=mock_result)):
        result = await classify_domains("Accused arrested for murder under BNS 103")
    assert isinstance(result, list)
    assert result[0]["domain"] == "criminal"
    assert result[0]["confidence"] == 0.92


@pytest.mark.asyncio
async def test_router_returns_cached_result_without_llm_call():
    from core.routing.domain_router import classify_domains
    from unittest.mock import AsyncMock, patch
    mock_redis = AsyncMock()
    mock_redis.get_router_result = AsyncMock(
        return_value=[{"domain": "criminal", "confidence": 0.95}]
    )
    with patch("core.routing.domain_router.get_redis", return_value=mock_redis):
        with patch("core.routing.domain_router.call_classifier") as mock_llm:
            result = await classify_domains("theft case", redis=mock_redis)
    mock_llm.assert_not_called()


def test_adapter_selector_returns_adapters_above_threshold():
    from core.routing.adapter_selector import select_adapters
    domains = [
        {"domain": "criminal", "confidence": 0.92},
        {"domain": "constitutional", "confidence": 0.45},
        {"domain": "tax", "confidence": 0.15},
    ]
    adapters = select_adapters(domains, threshold=0.40)
    assert "criminal" in adapters
    assert "constitutional" in adapters
    assert "tax" not in adapters


def test_adapter_selector_always_returns_at_least_one():
    from core.routing.adapter_selector import select_adapters
    domains = [{"domain": "criminal", "confidence": 0.10}]
    adapters = select_adapters(domains, threshold=0.40)
    assert adapters == ["criminal"]  # top-1 always included even below threshold
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_domain_router.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write core/routing/domain_router.py**

```python
# core/routing/domain_router.py
import json
from openai import AsyncOpenAI
from core.config import settings
from db.redis_client import RedisClient
from typing import Optional

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

DOMAINS = [
    "criminal", "constitutional", "civil", "corporate",
    "family", "property", "labour", "tax", "ip", "banking", "cyber"
]

ROUTER_PROMPT = f"""You are an Indian legal domain classifier.
Given a legal document summary or query, classify it into one or more of these domains:
{", ".join(DOMAINS)}

Return ONLY a valid JSON array like:
[{{"domain": "criminal", "confidence": 0.92}}, {{"domain": "constitutional", "confidence": 0.45}}]

Sort by confidence descending. Include only domains with confidence > 0.10.

Text to classify:
"""


async def call_classifier(text: str) -> list[dict]:
    response = await _client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": ROUTER_PROMPT + text[:2000]}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    # GPT returns JSON object, we need array inside it
    raw = json.loads(response.choices[0].message.content)
    if isinstance(raw, list):
        return raw
    # Handle {"domains": [...]} wrapper
    return raw.get("domains", raw.get("classifications", []))


async def classify_domains(
    text: str,
    redis: Optional[RedisClient] = None
) -> list[dict]:
    """Returns sorted list of {domain, confidence} dicts."""
    if redis:
        cached = await redis.get_router_result(text)
        if cached:
            return cached

    result = await call_classifier(text)

    if redis:
        await redis.set_router_result(text, result)

    return result
```

- [ ] **Step 4: Write core/routing/adapter_selector.py**

```python
# core/routing/adapter_selector.py
from core.config import settings


DOMAIN_TO_ADAPTER = {
    "criminal": "criminal",
    "constitutional": "constitutional",
    "civil": "civil",
    "corporate": "corporate",
    "family": "family",
    "property": "property",
    "labour": "labour",
    "tax": "tax",
    "ip": "ip",
    "banking": "banking",
    "cyber": "cyber",
}

# Adapters available on disk (populated as training completes)
AVAILABLE_ADAPTERS = {"criminal"}  # Start with IIT Patna criminal_v0 only


def select_adapters(
    domain_scores: list[dict],
    threshold: float = None,
) -> list[str]:
    """
    Return adapter names to activate, sorted by confidence descending.
    Always returns at least one adapter (top-1 regardless of threshold).
    Only activates adapters that exist in AVAILABLE_ADAPTERS.
    """
    if threshold is None:
        threshold = settings.SECONDARY_ADAPTER_CONFIDENCE

    sorted_domains = sorted(domain_scores, key=lambda x: x["confidence"], reverse=True)
    selected = []

    for item in sorted_domains:
        adapter = DOMAIN_TO_ADAPTER.get(item["domain"])
        if adapter and adapter in AVAILABLE_ADAPTERS:
            if item["confidence"] >= threshold or not selected:
                selected.append(adapter)

    # Fallback: if no available adapters matched, use criminal as default
    return selected if selected else ["criminal"]
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/unit/test_domain_router.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add core/routing/ tests/unit/test_domain_router.py
git commit -m "feat: domain router (zero-shot LLM) + adapter selector with Redis cache"
```

---

### Task 7: LoRA Engine — Load Base Model + Adapter Switching

**Files:**
- Create: `core/reasoning/lora_engine.py`
- Create: `tests/unit/test_lora_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_lora_engine.py
import pytest
from unittest.mock import MagicMock, patch


def test_lora_engine_loads_base_model_on_init():
    from core.reasoning.lora_engine import LoRAEngine
    with patch("core.reasoning.lora_engine.AutoModelForCausalLM") as mock_model_cls:
        with patch("core.reasoning.lora_engine.AutoTokenizer") as mock_tok_cls:
            mock_model_cls.from_pretrained.return_value = MagicMock()
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            engine = LoRAEngine.__new__(LoRAEngine)
            engine._model = None
            engine._tokenizer = None
            engine._loaded_adapters = set()
            engine._active_adapters = []
    assert engine._model is None  # lazy load not triggered yet


def test_lora_engine_select_adapters_updates_active():
    from core.reasoning.lora_engine import LoRAEngine
    engine = LoRAEngine.__new__(LoRAEngine)
    engine._model = MagicMock()
    engine._tokenizer = MagicMock()
    engine._loaded_adapters = {"criminal"}
    engine._active_adapters = []

    engine._model.set_adapter = MagicMock()
    engine._set_active_adapters(["criminal"])

    engine._model.set_adapter.assert_called_with(["criminal"])
    assert engine._active_adapters == ["criminal"]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_lora_engine.py -v
```

- [ ] **Step 3: Write core/reasoning/lora_engine.py**

```python
# core/reasoning/lora_engine.py
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from core.config import settings


class LoRAEngine:
    """
    Manages a single base model instance with hot-swappable LoRA adapters.
    Adapters are loaded lazily on first use and kept in memory.
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._loaded_adapters: set[str] = set()
        self._active_adapters: list[str] = []

    def _load_base_model(self):
        if self._model is not None:
            return
        print(f"Loading base model: {settings.BASE_MODEL_ID}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("Base model loaded.")

    def load_adapter(self, adapter_name: str):
        """Load a LoRA adapter from disk into the model. No-op if already loaded."""
        if adapter_name in self._loaded_adapters:
            return
        self._load_base_model()
        adapter_path = Path(settings.ADAPTERS_DIR) / adapter_name
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        if not self._loaded_adapters:
            self._model = PeftModel.from_pretrained(
                self._model, str(adapter_path), adapter_name=adapter_name
            )
        else:
            self._model.load_adapter(str(adapter_path), adapter_name=adapter_name)

        self._loaded_adapters.add(adapter_name)
        print(f"Adapter loaded: {adapter_name}")

    def _set_active_adapters(self, adapter_names: list[str]):
        self._model.set_adapter(adapter_names)
        self._active_adapters = adapter_names

    def activate(self, adapter_names: list[str]):
        """Load (if needed) and activate the given adapters."""
        for name in adapter_names:
            self.load_adapter(name)
        self._set_active_adapters(adapter_names)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        self._load_base_model()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt from the output
        return decoded[len(self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]


# Singleton — one model instance per process
lora_engine = LoRAEngine()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_lora_engine.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Load IIT Patna adapter as criminal_v0 baseline**

```bash
# scripts/seed_iit_patna_adapter.py
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="RMani1/indian-legal-qwen-lora",
    local_dir="adapters/criminal"
)
print("IIT Patna criminal_v0 adapter downloaded to adapters/criminal")
EOF
```

- [ ] **Step 6: Commit**

```bash
git add core/reasoning/lora_engine.py tests/unit/test_lora_engine.py scripts/seed_iit_patna_adapter.py
git commit -m "feat: LoRA engine with lazy base model load, hot-swap adapter activation via PEFT"
```

---

### Task 8: Citation Validator (Hallucination Guard)

**Files:**
- Create: `core/validation/citation_validator.py`
- Create: `tests/unit/test_citation_validator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_citation_validator.py
from core.validation.citation_validator import validate_citations, extract_citations


def test_extract_citations_finds_bns_references():
    text = "Under BNS Section 103, murder is punishable. Also see BNSS 187."
    citations = extract_citations(text)
    assert "BNS 103" in citations or "BNS Section 103" in citations
    assert len(citations) >= 1


def test_validate_citations_marks_grounded_as_verified():
    response = "The accused is charged under BNS 103 for murder."
    context = "BNS 103: Whoever commits murder shall be punished..."
    result = validate_citations(response, context)
    assert "[UNVERIFIED]" not in result


def test_validate_citations_flags_ungrounded_citation():
    response = "The accused is charged under BNS 999 for teleportation."
    context = "BNS 103: Whoever commits murder shall be punished..."
    result = validate_citations(response, context)
    assert "[UNVERIFIED]" in result


def test_validate_citations_does_not_modify_clean_response():
    response = "Based on the facts, the contract was breached."
    context = "Contract Act 1872 Section 73: compensation for breach..."
    result = validate_citations(response, context)
    assert result == response
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_citation_validator.py -v
```

- [ ] **Step 3: Write core/validation/citation_validator.py**

```python
# core/validation/citation_validator.py
import re


# Patterns for Indian legal citations
CITATION_PATTERNS = [
    r"BNS\s+(?:Section\s+)?\d+[A-Z]?",      # BNS 103, BNS Section 103
    r"BNSS\s+(?:Section\s+)?\d+[A-Z]?",     # BNSS 187
    r"BSA\s+(?:Section\s+)?\d+[A-Z]?",      # BSA 65
    r"IPC\s+(?:Section\s+)?\d+[A-Z]?",      # IPC 302 (legacy)
    r"Section\s+\d+[A-Z]?\s+of\s+the\s+\w+",
    r"Article\s+\d+[A-Z]?",                  # Article 21
]
_PATTERN = re.compile("|".join(CITATION_PATTERNS), re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    """Extract all legal citations from text."""
    return list(set(_PATTERN.findall(text)))


def _normalise(citation: str) -> str:
    """Normalise citation for comparison: lowercase, remove extra spaces."""
    return re.sub(r"\s+", " ", citation.lower().strip())


def validate_citations(response: str, context: str) -> str:
    """
    For every citation in response, check if it appears in context.
    Citations not found in context are flagged with [UNVERIFIED].
    Returns the response string with [UNVERIFIED] tags appended to bad citations.
    """
    citations = extract_citations(response)
    if not citations:
        return response

    context_normalised = _normalise(context)
    result = response

    for citation in citations:
        if _normalise(citation) not in context_normalised:
            result = result.replace(citation, f"{citation} [UNVERIFIED]")

    return result
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_citation_validator.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add core/validation/ tests/unit/test_citation_validator.py
git commit -m "feat: citation validator — flags ungrounded section citations as [UNVERIFIED]"
```

---

### Task 9: Case Research Pipeline

**Files:**
- Create: `core/reasoning/context_merger.py`
- Create: `core/reasoning/case_research.py`
- Create: `tests/unit/test_case_research.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_case_research.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_case_research_returns_answer_with_citations():
    from core.reasoning.case_research import run_case_research
    mock_engine = MagicMock()
    mock_engine.activate = MagicMock()
    mock_engine.generate = MagicMock(return_value="BNS 103 applies here. The accused faces murder charges.")

    with patch("core.reasoning.case_research.lora_engine", mock_engine):
        with patch("core.reasoning.case_research.get_merged_context",
                   new=AsyncMock(return_value="BNS 103: murder section...")):
            result = await run_case_research(
                query="What section applies for murder?",
                case_id="CASE-001",
                adapters=["criminal"],
                session_history=[],
            )
    assert "answer" in result
    assert "context_used" in result
    assert "adapters_used" in result


@pytest.mark.asyncio
async def test_case_research_passes_conversation_history():
    from core.reasoning.case_research import build_research_prompt
    history = [
        {"role": "user", "content": "What is BNS?"},
        {"role": "assistant", "content": "BNS is Bharatiya Nyaya Sanhita."},
    ]
    prompt = build_research_prompt("Follow up question", "legal context here", history)
    assert "BNS is Bharatiya Nyaya Sanhita" in prompt
    assert "Follow up question" in prompt
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_case_research.py -v
```

- [ ] **Step 3: Write core/reasoning/context_merger.py**

```python
# core/reasoning/context_merger.py
from core.indexing.pageindex_query import get_toc_summary, fetch_node_by_id
from db.redis_client import RedisClient
from db.mongo import mongo
from typing import Optional


async def get_merged_context(
    query: str,
    case_id: str,
    redis: RedisClient,
    max_legal_nodes: int = 5,
) -> str:
    """
    Merge relevant context from:
    1. Client case PageIndex (fetched from Redis or MongoDB)
    2. Legal DB PageIndex (fetched from Redis or MongoDB)
    Returns a combined context string for LLM consumption.
    """
    # Client PageIndex
    case_tree = await redis.get_case_pageindex(case_id)
    if not case_tree:
        case_tree = await mongo.case_indexes.find_one({"case_id": case_id})

    client_toc = get_toc_summary(case_tree, max_depth=2) if case_tree else "No client documents indexed."

    # Legal DB: search for relevant nodes by keyword match
    keywords = query.lower().split()[:5]
    legal_nodes = await mongo.legal_nodes.find(
        {"keywords": {"$in": keywords}},
        limit=max_legal_nodes,
    ).to_list(length=max_legal_nodes)

    legal_context = "\n\n".join(
        f"[{n['node_id']}] {n['title']}\n{n.get('text', n.get('summary', ''))}"
        for n in legal_nodes
    ) if legal_nodes else "No matching legal sections found."

    return f"""=== CLIENT DOCUMENTS ===
{client_toc}

=== RELEVANT LEGAL SECTIONS ===
{legal_context}
"""
```

- [ ] **Step 4: Write core/reasoning/case_research.py**

```python
# core/reasoning/case_research.py
from core.reasoning.lora_engine import lora_engine
from core.reasoning.context_merger import get_merged_context
from core.validation.citation_validator import validate_citations
from db.redis_client import RedisClient

RESEARCH_SYSTEM = """You are an expert Indian legal research assistant.
Answer the lawyer's query using ONLY the provided legal context.
Cite section numbers exactly as they appear in the context.
Do not cite sections from memory. Be precise and structured."""


def build_research_prompt(
    query: str,
    context: str,
    history: list[dict],
) -> str:
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
    )
    return f"""{RESEARCH_SYSTEM}

=== CONTEXT ===
{context}

=== CONVERSATION HISTORY ===
{history_text}

=== CURRENT QUERY ===
LAWYER: {query}

ASSISTANT:"""


async def run_case_research(
    query: str,
    case_id: str,
    adapters: list[str],
    session_history: list[dict],
    redis: RedisClient,
) -> dict:
    context = await get_merged_context(query, case_id, redis)
    prompt = build_research_prompt(query, context, session_history)

    lora_engine.activate(adapters)
    raw_answer = lora_engine.generate(prompt, max_new_tokens=512, temperature=0.3)

    validated_answer = validate_citations(raw_answer, context)

    return {
        "answer": validated_answer,
        "context_used": context,
        "adapters_used": adapters,
        "query_type": "research",
    }
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/unit/test_case_research.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add core/reasoning/ tests/unit/test_case_research.py
git commit -m "feat: case research pipeline — context merger, LoRA generation, citation validation"
```

---

### Task 10: FastAPI Backend + Gradio UI

**Files:**
- Create: `api/main.py`
- Create: `api/deps.py`
- Create: `api/routes/cases.py`
- Create: `api/routes/documents.py`
- Create: `api/routes/query.py`
- Create: `ui/app.py`

- [ ] **Step 1: Write api/deps.py**

```python
# api/deps.py
from fastapi import Header, HTTPException, status
from db.mongo import mongo
from db.redis_client import RedisClient
from functools import lru_cache

_redis: RedisClient = None


def get_redis() -> RedisClient:
    global _redis
    if _redis is None:
        _redis = RedisClient()
    return _redis


async def require_api_key(x_api_key: str = Header(...)) -> str:
    # Phase 1: simple static key from env
    from core.config import settings
    if not hasattr(settings, "API_KEY") or x_api_key != getattr(settings, "API_KEY", "dev-key"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return x_api_key
```

- [ ] **Step 2: Write api/main.py**

```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from db.mongo import mongo
from db.postgres import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongo.connect()
    await init_db()
    yield
    await mongo.disconnect()


app = FastAPI(title="Indian Legal AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import cases, documents, query
app.include_router(cases.router, prefix="/cases", tags=["cases"])
app.include_router(documents.router, prefix="/cases", tags=["documents"])
app.include_router(query.router, prefix="/cases", tags=["query"])


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 3: Write api/routes/query.py**

```python
# api/routes/query.py
import time
import uuid
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.deps import get_redis, require_api_key
from core.routing.domain_router import classify_domains
from core.routing.adapter_selector import select_adapters
from core.reasoning.case_research import run_case_research
from db.redis_client import RedisClient
from db.postgres import AsyncSessionLocal, AuditLog

router = APIRouter()


class QueryRequest(BaseModel):
    session_id: str
    query: str
    query_type: str = "research"  # research | qa | argument


class QueryResponse(BaseModel):
    answer: str
    adapters_used: list[str]
    latency_ms: float
    session_id: str


@router.post("/{case_id}/query", response_model=QueryResponse)
async def query_case(
    case_id: str,
    req: QueryRequest,
    redis: RedisClient = Depends(get_redis),
    _: str = Depends(require_api_key),
):
    start = time.time()

    # Get conversation history from Redis
    history = await redis.get_session_history(req.session_id)

    # Classify domain + select adapters
    domain_scores = await classify_domains(req.query, redis=redis)
    adapters = select_adapters(domain_scores)

    # Run pipeline
    result = await run_case_research(
        query=req.query,
        case_id=case_id,
        adapters=adapters,
        session_history=history,
        redis=redis,
    )

    # Persist to session history
    await redis.append_session_message(req.session_id, {"role": "user", "content": req.query})
    await redis.append_session_message(req.session_id, {"role": "assistant", "content": result["answer"]})

    latency_ms = (time.time() - start) * 1000

    # Audit log
    async with AsyncSessionLocal() as db:
        db.add(AuditLog(
            case_id=case_id,
            session_id=req.session_id,
            query_type=req.query_type,
            query=req.query,
            adapters_used=adapters,
            latency_ms=latency_ms,
        ))
        await db.commit()

    return QueryResponse(
        answer=result["answer"],
        adapters_used=adapters,
        latency_ms=round(latency_ms, 2),
        session_id=req.session_id,
    )
```

- [ ] **Step 4: Write ui/app.py**

```python
# ui/app.py
import gradio as gr
import httpx
import uuid

API_URL = "http://api:8000"
API_KEY = "dev-key"


def chat(message: str, history: list, case_id: str, session_id: str):
    if not case_id.strip():
        return history + [[message, "Please enter a Case ID first."]], session_id

    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        response = httpx.post(
            f"{API_URL}/cases/{case_id}/query",
            json={"session_id": session_id, "query": message, "query_type": "research"},
            headers={"x-api-key": API_KEY},
            timeout=60,
        )
        answer = response.json()["answer"]
    except Exception as e:
        answer = f"Error: {e}"

    return history + [[message, answer]], session_id


with gr.Blocks(title="Indian Legal AI Assistant") as demo:
    gr.Markdown("# Indian Legal AI Assistant")
    gr.Markdown("Enter your Case ID and ask questions about your case or Indian law.")

    with gr.Row():
        case_id_box = gr.Textbox(label="Case ID", placeholder="CASE-2024-001")

    chatbot = gr.Chatbot(height=500)
    session_state = gr.State("")
    msg_box = gr.Textbox(label="Your query", placeholder="What sections apply for cheating in this FIR?")
    send_btn = gr.Button("Send", variant="primary")

    send_btn.click(
        chat,
        inputs=[msg_box, chatbot, case_id_box, session_state],
        outputs=[chatbot, session_state],
    )
    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot, case_id_box, session_state],
        outputs=[chatbot, session_state],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 5: Start full stack and smoke test**

```bash
docker compose -f docker/docker-compose.yml up -d
sleep 10
curl http://localhost:8000/health
```

Expected: `{"status": "ok"}`

```bash
curl -X POST http://localhost:8000/cases/CASE-001/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{"session_id": "test-1", "query": "What is BNS Section 103?", "query_type": "research"}'
```

Expected: JSON with `answer`, `adapters_used`, `latency_ms`.

- [ ] **Step 6: Commit**

```bash
git add api/ ui/ tests/
git commit -m "feat: FastAPI backend + Gradio UI, end-to-end case research pipeline live"
```

---

### Task 10b: Case + Document Management Routes

**Files:**
- Create: `api/routes/cases.py`
- Create: `api/routes/documents.py`

- [ ] **Step 1: Write api/routes/cases.py**

```python
# api/routes/cases.py
import uuid
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from db.postgres import AsyncSessionLocal, Case
from api.deps import require_api_key

router = APIRouter()


class CreateCaseRequest(BaseModel):
    title: str
    lawyer_id: str


@router.post("", status_code=201)
async def create_case(req: CreateCaseRequest, _: str = Depends(require_api_key)):
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    async with AsyncSessionLocal() as db:
        db.add(Case(id=case_id, title=req.title, lawyer_id=req.lawyer_id))
        await db.commit()
    return {"case_id": case_id, "title": req.title}


@router.get("/{case_id}")
async def get_case(case_id: str, _: str = Depends(require_api_key)):
    async with AsyncSessionLocal() as db:
        case = await db.get(Case, case_id)
        if not case:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Case not found")
        return {"case_id": case.id, "title": case.title, "domains": case.domains}
```

- [ ] **Step 2: Write api/routes/documents.py**

```python
# api/routes/documents.py
import uuid
from fastapi import APIRouter, Depends, UploadFile, File
from api.deps import get_redis, require_api_key
from core.ingestion.pdf_loader import load_pdf_from_bytes
from core.ingestion.summarizer import summarize_document
from core.ingestion.doc_classifier import normalise_doc_type
from core.indexing.pageindex_builder import build_case_pageindex
from db.redis_client import RedisClient
from db.mongo import mongo
from db.postgres import AsyncSessionLocal, Document, Case

router = APIRouter()


@router.post("/{case_id}/documents", status_code=201)
async def upload_document(
    case_id: str,
    file: UploadFile = File(...),
    redis: RedisClient = Depends(get_redis),
    _: str = Depends(require_api_key),
):
    doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
    raw_bytes = await file.read()

    # Load pages
    pages = load_pdf_from_bytes(raw_bytes)

    # Extract text from each page (basic — page images handled in vision mode)
    for page in pages:
        page["text"] = ""  # Phase 1: OCR-free, use LLM on image in Phase 2

    # Summarise document
    combined_text = " ".join(p.get("text", "") for p in pages[:3])
    summary = await summarize_document(combined_text or file.filename)
    doc_type = normalise_doc_type(summary.get("doc_type", "other"))

    # Build + cache PageIndex
    await redis.invalidate_case_pageindex(case_id)  # invalidate old cache
    tree = await build_case_pageindex(case_id, doc_id, doc_type, pages)
    await mongo.case_indexes.replace_one(
        {"case_id": case_id, "doc_id": doc_id}, tree, upsert=True
    )
    await redis.set_case_pageindex(case_id, tree)

    # Store metadata in Postgres
    async with AsyncSessionLocal() as db:
        db.add(Document(
            id=doc_id, case_id=case_id, doc_type=doc_type,
            filename=file.filename, page_count=len(pages),
            storage_path=f"{case_id}/{doc_id}/{file.filename}",
        ))
        await db.commit()

    return {"doc_id": doc_id, "doc_type": doc_type, "pages": len(pages)}
```

- [ ] **Step 3: Commit**

```bash
git add api/routes/cases.py api/routes/documents.py
git commit -m "feat: case management and document upload routes with PageIndex build + cache"
```

---

### Task 11: InLegalBench Zero-Shot Baseline Measurement

**Files:**
- Create: `tests/eval/run_inlegalbench.py`

- [ ] **Step 1: Write evaluation script**

```python
# tests/eval/run_inlegalbench.py
"""
Run InLegalBench evaluation against current pipeline.
Usage: python tests/eval/run_inlegalbench.py --dataset data/eval/inlegalbench_sample.json

Expected output: accuracy score + per-category breakdown.
This baseline MUST be recorded before any LoRA adapter training begins.
It establishes whether LoRA training is worth the 6-week data prep investment.

Decision rule:
  - If accuracy >= 65% → LoRA is optional enhancement
  - If accuracy < 55% → LoRA is necessary, proceed with adapter training
"""
import asyncio
import json
import argparse
from pathlib import Path


async def evaluate(dataset_path: str) -> dict:
    from core.routing.domain_router import classify_domains
    from core.routing.adapter_selector import select_adapters
    from core.reasoning.case_research import run_case_research
    from db.redis_client import RedisClient

    dataset = json.loads(Path(dataset_path).read_text())
    redis = RedisClient()
    correct = 0
    results_by_domain = {}

    for i, item in enumerate(dataset):
        query = item["question"]
        expected = item["answer"].strip().lower()
        domain = item.get("domain", "unknown")

        domain_scores = await classify_domains(query, redis=redis)
        adapters = select_adapters(domain_scores)

        result = await run_case_research(
            query=query,
            case_id="EVAL-CASE",
            adapters=adapters,
            session_history=[],
            redis=redis,   # always passed — never None
        )
        predicted = result["answer"].strip().lower()
        is_correct = expected in predicted or predicted in expected

        if is_correct:
            correct += 1
        results_by_domain.setdefault(domain, {"correct": 0, "total": 0})
        results_by_domain[domain]["total"] += 1
        if is_correct:
            results_by_domain[domain]["correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(dataset)}, running accuracy: {correct/(i+1):.2%}")

    accuracy = correct / len(dataset)
    print(f"\n=== InLegalBench Results ===")
    print(f"Overall accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")
    print("\nPer-domain breakdown:")
    for domain, stats in results_by_domain.items():
        acc = stats["correct"] / stats["total"]
        print(f"  {domain}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    await redis.close()
    return {"accuracy": accuracy, "by_domain": results_by_domain}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to InLegalBench JSON file")
    args = parser.parse_args()
    asyncio.run(evaluate(args.dataset))
```

- [ ] **Step 2: Run baseline and RECORD the result**

```bash
python tests/eval/run_inlegalbench.py --dataset data/eval/inlegalbench_sample.json
```

**Record the accuracy score in `docs/superpowers/specs/baseline_results.md`.** This number drives the Phase 2 go/no-go decision on LoRA training.

- [ ] **Step 3: Commit**

```bash
git add tests/eval/run_inlegalbench.py
git commit -m "eval: InLegalBench zero-shot baseline measurement script"
```

---

## Phase 2 — Domain Adapters (Weeks 5–12)

**Deliverable:** All 11 domain LoRA adapters trained, benchmarked, and integrated. BERT router replacing zero-shot. Multi-adapter composition tested and validated.

**Go/No-Go gate:** Only proceed if Phase 1 baseline < 65% on InLegalBench.

### Key tasks (each follows same TDD pattern as Phase 1):

**Task 12: Data Pipeline — Scraping + Formatting**
- `training/data_prep/kanoon_scraper.py` — async scraper for Indian Kanoon judgments
- `training/data_prep/format_qa_pairs.py` — convert judgments to instruction-tuning JSONL format:
  ```json
  {"system": "You are an expert Indian criminal law assistant.",
   "user": "What does Section 103 BNS say about murder?",
   "assistant": "Section 103 BNS (equivalent to IPC 302) states that..."}
  ```
- `training/data_prep/domain_splitter.py` — split dataset by legal domain tag
- Target: 3,000–5,000 pairs per domain, deduplicated, human spot-checked

**Task 13: QLoRA Training Script**
- `training/train_adapter.py` — single script, config-driven:
  ```bash
  python training/train_adapter.py --config training/configs/criminal.yaml
  ```
- `training/configs/base.yaml`:
  ```yaml
  base_model: Qwen/Qwen2.5-7B-Instruct
  output_dir: adapters/{domain}
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  load_in_4bit: true
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  lr_scheduler_type: cosine
  ```
- Criminal config overrides: same as base (r=32)
- Argument gen config overrides: `lora_r: 128, lora_alpha: 256, target_modules: [q,k,v,o,gate,up,down]`

**Task 14: Base Model Comparison**
- Train Criminal adapter on Qwen2.5-7B, Llama-3.1-8B, Mistral-7B
- Run `training/eval_adapter.py` against InLegalBench for each
- Pick winner → use for all subsequent adapter training
- Write result to `docs/superpowers/specs/base_model_comparison.md`

**Task 15: BERT Domain Router (Phase 2)**
- Replace zero-shot LLM router with fine-tuned `legal-bert-base-uncased` classifier
- Train on labelled domain data collected from Phase 1 usage logs (PostgreSQL audit table)
- Target: > 0.85 multi-label F1, < 50ms inference
- `core/routing/domain_router.py` gets a `BERTRouter` class alongside existing `ZeroShotRouter`
- Feature flag in config: `ROUTER_TYPE=zero_shot|bert`

**Task 16: Multi-Adapter Composition Test**
- Run InLegalBench on: single adapter vs 2 adapters vs 3 adapters
- If accuracy drop > 5% for 2-adapter case: pre-merge top-3 common combinations using `add_weighted_adapter()`
- Document result in spec TODO-M3

---

## Phase 3 — Argument Generation (Weeks 13–20)

**Deliverable:** Full argument generation pipeline with CoT, agentic tool calls, critic scoring, and top-K filtering.

### Key tasks:

**Task 17: Agentic Tool Definitions**
- `core/tools/fetch_legal_db.py`:
  ```python
  async def fetch_legal_db(node_id: str, redis: RedisClient, mongo) -> str:
      node = await redis.get_legal_node(node_id)
      if not node:
          node = await mongo.legal_nodes.find_one({"node_id": node_id})
      return node["text"] if node else f"Node {node_id} not found."
  ```
- `core/tools/resolve_cross_ref.py` — parse "Section 103 BNS" → look up node_id → fetch content
- `core/tools/search_precedents.py` — keyword search on legal_nodes collection + case_indexes

**Task 18: CoT Argument Generation Pipeline**
- `core/reasoning/argument_gen.py`
- Stage 1: small model + domain LoRA → structured legal brief (JSON output)
- Stage 2: GPT-4.1 API with function calling tools, 6-step CoT prompt
- Agentic loop: model continues CoT after each tool response
- Generates K=5 candidates

**Task 19: Critic Model (Phase 1 — GPT-4.1 as Judge)**
- `core/critic/scorer.py` — sends argument + rubric to GPT-4.1
- `core/critic/filter.py` — sorts by score, applies threshold=0.72, retries at temp=0.9 if all fail

**Task 20: Argument Gen LoRA Adapter**
- Same training pipeline as Task 13 but with `argument_gen.yaml` config (r=128)
- Training data: 3,000 legal brief examples (written arguments, pleadings from Indian courts)
- Integration: `lora_engine.activate(["criminal", "argument-gen"])` in Stage 1

---

## Phase 4 — Hardening (Weeks 21+)

**Deliverable:** Production-ready system with Phase 2 critic, on-prem option, security audit.

### Key tasks:

**Task 21: Critic Reward Model (Phase 2)**
- Collect human lawyer ratings from Phase 3 usage (stored in PostgreSQL `argument_ratings` table)
- Train `lora-critic` adapter on rated argument pairs
- Replace GPT-4.1 judge call in `core/critic/scorer.py` with local model inference
- Cost savings: ~$0.02 → ~$0.001 per argument evaluation

**Task 22: On-Prem Upgrade Path**
- Add `GENERATION_BACKEND=api|local` config flag
- When `local`: swap GPT-4.1 API calls in `argument_gen.py` for a quantised 70B local model
- `core/reasoning/llm_client.py` abstraction layer so swap is one-line config change

**Task 23: Performance + Latency**
- Profile end-to-end latency per query type
- Target: case research < 5s, doc Q&A < 3s
- Bottlenecks: model loading (mitigate with pre-warm on startup), MongoDB queries (add indexes on `domain` + `keywords` compound index)

**Task 24: Security + Privacy Audit**
- Client documents never logged in plaintext (only case_id + doc_id in audit logs)
- Redis TTLs enforced (no indefinite client data retention)
- API key rotation mechanism
- Disclaimer system: every response includes "This is not a substitute for professional legal advice."
- Rate limiting validated under load

---

## Testing Strategy Summary

| Layer | Tool | When |
|-------|------|------|
| Unit tests | pytest + pytest-mock | Every task, TDD |
| Integration tests | pytest-asyncio + real Docker services | End of each phase |
| Eval / benchmark | Custom eval scripts | Phase 1 baseline, Phase 2 after each adapter |
| Load testing | locust (add in Phase 4) | Before production |

Run all tests:
```bash
pytest tests/unit/ -v                     # fast, mocked
pytest tests/integration/ -v              # requires docker services running
python tests/eval/run_inlegalbench.py --dataset data/eval/inlegalbench_sample.json
```
