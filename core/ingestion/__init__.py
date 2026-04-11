# core/ingestion — document intake: OCR, summarisation, classification
from core.ingestion.ocr_module import load_ocr_model, ocr_file, clean_ocr_text
from core.ingestion.legal_bert_pipeline import (
    load_summarization_models,
    run_local_summarization,
    extractive_summary,
    extract_entities,
    preprocess_text,
)
from core.ingestion.summarizer import summarize_document, load_local_models
from core.ingestion.doc_classifier import normalise_doc_type
from core.ingestion.pdf_loader import load_pdf_pages, load_pdf_from_bytes

__all__ = [
    "load_ocr_model", "ocr_file", "clean_ocr_text",
    "load_summarization_models", "run_local_summarization",
    "extractive_summary", "extract_entities", "preprocess_text",
    "summarize_document", "load_local_models", "normalise_doc_type",
    "load_pdf_pages", "load_pdf_from_bytes",
]
