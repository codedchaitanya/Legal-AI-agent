"""
Hybrid summariser: LOCAL pipeline (Legal-BERT + PEGASUS) as primary,
Claude API as fallback for when local models aren't loaded or fail.

Public API:
    load_local_models()           → model bundle (call once at startup)
    summarize_document(text, ...) → structured summary dict
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Optional

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPC → BNS mapping (shared across both paths)
# ---------------------------------------------------------------------------
IPC_TO_BNS = {
    "302": "BNS 103", "304": "BNS 105", "307": "BNS 109",
    "376": "BNS 63",  "420": "BNS 318", "379": "BNS 303",
    "498A": "BNS 85", "406": "BNS 316", "354": "BNS 74",
    "306": "BNS 108", "304A": "BNS 106", "323": "BNS 115",
    "363": "BNS 137", "364": "BNS 138", "365": "BNS 139",
    "392": "BNS 309", "395": "BNS 310", "120B": "BNS 61",
}

# ---------------------------------------------------------------------------
# Local model management (lazy singleton)
# ---------------------------------------------------------------------------
_local_models: dict[str, Any] | None = None


def load_local_models() -> dict[str, Any]:
    """Load Legal-BERT + PEGASUS + spaCy. Cached after first call."""
    global _local_models
    if _local_models is not None:
        return _local_models
    try:
        from core.ingestion.legal_bert_pipeline import load_summarization_models
        _local_models = load_summarization_models()
        logger.info("Local summarisation models loaded (Legal-BERT + PEGASUS).")
        return _local_models
    except Exception as e:
        logger.warning("Failed to load local models: %s — will use Claude fallback.", e)
        return {}


def _local_summarize(text: str) -> dict | None:
    """Try local Legal-BERT + PEGASUS pipeline. Returns None on failure."""
    models = load_local_models()
    if not models:
        return None
    try:
        from core.ingestion.legal_bert_pipeline import run_local_summarization
        result = run_local_summarization(text, models)
        return {
            "summary": result["abstractive_summary"] or result["extractive_summary"],
            "extractive_summary": result["extractive_summary"],
            "entities": result.get("entities", {}),
            "sections_mentioned": result.get("sections_mentioned", []),
            "doc_type": "other",
            "keywords": [],
            "source": "local",
        }
    except Exception as e:
        logger.warning("Local summarisation failed: %s — falling back to Claude.", e)
        return None


# ---------------------------------------------------------------------------
# Claude API fallback
# ---------------------------------------------------------------------------

_anthropic_client = None

def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic_client


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


async def _claude_summarize(text: str) -> dict:
    """Claude Haiku fallback summariser."""
    client = _get_anthropic_client()
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": SUMMARIZE_PROMPT + text[:8000]}],
        temperature=0,
    )
    raw = response.content[0].text
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned invalid JSON: {raw[:200]}") from exc

    for sec in result.get("sections_mentioned", []):
        raw_sec = sec.get("raw", "")
        for ipc_num, bns_equiv in IPC_TO_BNS.items():
            if re.search(rf'\b{re.escape(ipc_num)}\b', raw_sec) and not sec.get("bns_equivalent"):
                sec["bns_equivalent"] = bns_equiv
    result["source"] = "claude"
    return result


# ---------------------------------------------------------------------------
# OCR-aware text cleaning
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """Extended preprocessing for OCR output before summarisation."""
    if not text or not isinstance(text, str):
        return ""
    t = text
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2014", " - ").replace("\u2013", "-")
    t = t.replace("\u00a0", " ")
    t = re.sub(r"(?m)^\s*(?:Page\s+)?\d+\s*(?:of\s+\d+)?\s*$", "", t)
    t = re.sub(r"(?m)^[\-_=\.~\*]{4,}\s*$", "", t)

    lines = t.splitlines()
    line_counts = Counter(ln.strip() for ln in lines if ln.strip())
    boilerplate = {ln for ln, cnt in line_counts.items() if cnt >= 3 and len(ln) < 120}
    if boilerplate:
        lines = [ln for ln in lines if ln.strip() not in boilerplate]
        t = "\n".join(lines)

    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def summarize_document(
    text: str,
    ocr_input: bool = False,
    force_claude: bool = False,
) -> dict:
    """
    Summarise a legal document.

    Strategy: try local Legal-BERT + PEGASUS first (free, fast).
    Fall back to Claude API if local models aren't available or fail.
    Set force_claude=True to skip local and go straight to API.
    """
    if not text or not text.strip():
        return {"summary": "", "entities": {}, "sections_mentioned": [], "doc_type": "other", "keywords": [], "source": "none"}

    cleaned = clean_ocr_text(text) if ocr_input else text

    if not force_claude:
        local_result = _local_summarize(cleaned)
        if local_result and local_result.get("summary"):
            return local_result
        logger.info("Local summarisation unavailable or empty — using Claude fallback.")

    return await _claude_summarize(cleaned)
