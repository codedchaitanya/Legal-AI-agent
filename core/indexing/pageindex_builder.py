# core/indexing/pageindex_builder.py
from anthropic import AsyncAnthropic
from core.config import settings

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


PAGE_SUMMARY_PROMPT = """Summarise this legal document page in 1-2 sentences.
Focus on: what legal event/content is described, any section numbers mentioned, key parties.
Return ONLY the summary sentence(s), nothing else.

Page text:
"""


async def generate_page_summary(page_text: str) -> str:
    response = await _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": PAGE_SUMMARY_PROMPT + page_text[:3000]}],
        temperature=0,
    )
    return response.content[0].text.strip()


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
