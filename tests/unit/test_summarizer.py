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


@pytest.mark.asyncio
async def test_summarizer_enriches_missing_bns_equivalent():
    from core.ingestion.summarizer import summarize_document
    mock_response = {
        "summary": "FIR filed against accused for theft",
        "entities": {"persons": ["Raju"], "dates": ["2024-01-15"]},
        "sections_mentioned": [{"raw": "Section 379 IPC"}],  # no bns_equivalent
        "doc_type": "FIR",
        "keywords": ["theft", "FIR", "accused"]
    }
    with patch("core.ingestion.summarizer.call_llm", new=AsyncMock(return_value=mock_response)):
        result = await summarize_document("Some FIR text here")
        assert result["sections_mentioned"][0]["bns_equivalent"] == "BNS 303"
