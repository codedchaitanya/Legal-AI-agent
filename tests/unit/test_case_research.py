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
                redis=AsyncMock(),
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
