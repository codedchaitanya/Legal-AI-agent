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
