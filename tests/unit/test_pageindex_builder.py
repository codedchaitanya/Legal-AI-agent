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
