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
