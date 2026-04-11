# core/ingestion/doc_classifier.py
# Thin wrapper — doc_type is already returned by summarizer.
# Extension point for Phase 2 BERT-based classifier.

VALID_DOC_TYPES = frozenset([
    "FIR", "contract", "affidavit", "judgment",
    "chargesheet", "notice", "other"
])


def normalise_doc_type(raw_type: str) -> str:
    """Normalise LLM-returned doc_type to a known enum value."""
    cleaned = raw_type.strip().lower()
    mapping = {
        "fir": "FIR", "first information report": "FIR",
        "contract": "contract", "agreement": "contract",
        "affidavit": "affidavit", "sworn statement": "affidavit",
        "judgment": "judgment", "order": "judgment", "verdict": "judgment",
        "chargesheet": "chargesheet", "charge sheet": "chargesheet",
        "notice": "notice", "legal notice": "notice",
    }
    return mapping.get(cleaned, "other")
