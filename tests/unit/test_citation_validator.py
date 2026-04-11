# tests/unit/test_citation_validator.py
from core.validation.citation_validator import validate_citations, extract_citations


def test_extract_citations_finds_bns_references():
    text = "Under BNS Section 103, murder is punishable. Also see BNSS 187."
    citations = extract_citations(text)
    assert "BNS 103" in citations or "BNS Section 103" in citations
    assert len(citations) >= 1


def test_validate_citations_marks_grounded_as_verified():
    response = "The accused is charged under BNS 103 for murder."
    context = "BNS 103: Whoever commits murder shall be punished..."
    result = validate_citations(response, context)
    assert "[UNVERIFIED]" not in result


def test_validate_citations_flags_ungrounded_citation():
    response = "The accused is charged under BNS 999 for teleportation."
    context = "BNS 103: Whoever commits murder shall be punished..."
    result = validate_citations(response, context)
    assert "[UNVERIFIED]" in result


def test_validate_citations_does_not_modify_clean_response():
    response = "Based on the facts, the contract was breached."
    context = "Contract Act 1872 Section 73: compensation for breach..."
    result = validate_citations(response, context)
    assert result == response
