# core/validation/citation_validator.py
"""Post-generation hallucination guard for Indian legal citations."""
import re


CITATION_PATTERNS = [
    r"BNS\s+(?:Section\s+)?\d+[A-Z]?",
    r"BNSS\s+(?:Section\s+)?\d+[A-Z]?",
    r"BSA\s+(?:Section\s+)?\d+[A-Z]?",
    r"IPC\s+(?:Section\s+)?\d+[A-Z]?",
    r"CrPC\s+(?:Section\s+)?\d+[A-Z]?",
    r"Section\s+\d+[A-Z]?\s+of\s+the\s+\w+",
    r"Article\s+\d+[A-Z]?",
    r"Order\s+[IVXLC0-9]+\s+Rule\s+\d+",
]
_PATTERN = re.compile("|".join(CITATION_PATTERNS), re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    """Extract all legal citations from text."""
    return list(set(_PATTERN.findall(text)))


def _normalise(citation: str) -> str:
    return re.sub(r"\s+", " ", citation.lower().strip())


def validate_citations(response: str, context: str) -> str:
    """
    Verify every citation in the response appears in the provided context.
    Unverified citations are tagged with [UNVERIFIED].
    """
    citations = extract_citations(response)
    if not citations:
        return response

    context_normalised = _normalise(context)
    result = response

    for citation in citations:
        if _normalise(citation) not in context_normalised:
            result = result.replace(citation, f"{citation} [UNVERIFIED]")

    return result
