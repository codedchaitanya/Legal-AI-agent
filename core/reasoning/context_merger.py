# core/reasoning/context_merger.py
from core.indexing.pageindex_query import get_toc_summary
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
