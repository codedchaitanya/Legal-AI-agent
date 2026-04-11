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

A:"""


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
