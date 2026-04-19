# api/routes/query.py
import time
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.deps import get_redis, require_api_key
from core.routing.domain_router import classify_domains
from core.routing.adapter_selector import select_adapters
from core.reasoning.case_research import run_case_research
from core.reasoning.argument_generator import generate_arguments
from db.redis_client import RedisClient
from db.postgres import AsyncSessionLocal, AuditLog

router = APIRouter()


class QueryRequest(BaseModel):
    session_id: str
    query: str
    query_type: str = "research"  # "research" | "arguments"
    side: str = "both"            # for query_type="arguments": "prosecution"|"defense"|"both"
    image_paths: list[str] = []   # optional court exhibit images


class QueryResponse(BaseModel):
    answer: str | None = None
    arguments: dict | None = None
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

    history = await redis.get_session_history(req.session_id)
    domain_scores = await classify_domains(req.query, redis=redis)
    adapters = select_adapters(domain_scores)
    primary_domain = adapters[0] if adapters else "civil_general"

    result = await run_case_research(
        query=req.query,
        case_id=case_id,
        adapters=adapters,
        session_history=history,
        redis=redis,
    )

    arguments = None
    if req.query_type == "arguments":
        # Fetch case summary from Redis (set during document ingestion)
        case_summary = await redis.get(f"summary:{case_id}") or result["answer"]
        arguments = await generate_arguments(
            case_summary=case_summary,
            research_context=result["context_used"],
            domain=primary_domain,
            image_paths=req.image_paths or [],
            side=req.side,
        )

    await redis.append_session_message(req.session_id, {"role": "user", "content": req.query})
    await redis.append_session_message(req.session_id, {"role": "assistant", "content": result["answer"]})

    latency_ms = (time.time() - start) * 1000

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
        arguments=arguments,
        adapters_used=adapters,
        latency_ms=round(latency_ms, 2),
        session_id=req.session_id,
    )
