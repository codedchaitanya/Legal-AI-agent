# core/indexing/pageindex_query.py
from typing import Optional


def fetch_node_by_id(tree: dict, node_id: str) -> Optional[dict]:
    """Recursively search tree for a node by node_id."""
    if tree.get("node_id") == node_id:
        return tree
    for child in tree.get("sub_nodes", []):
        result = fetch_node_by_id(child, node_id)
        if result:
            return result
    return None


def get_toc_summary(tree: dict, max_depth: int = 2) -> str:
    """
    Return a flat text Table of Contents for LLM consumption.
    Format: [node_id] title — summary
    """
    lines = []

    def _walk(node: dict, depth: int):
        if depth > max_depth:
            return
        indent = "  " * depth
        lines.append(f"{indent}[{node['node_id']}] {node.get('title', '')} — {node.get('summary', '')}")
        for child in node.get("sub_nodes", []):
            _walk(child, depth + 1)

    _walk(tree, 0)
    return "\n".join(lines)
