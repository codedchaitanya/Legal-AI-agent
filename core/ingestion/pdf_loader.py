# core/ingestion/pdf_loader.py
from pdf2image import convert_from_path
from pathlib import Path
import tempfile
import os


def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Convert PDF to list of page dicts.
    Each dict: {page_number: int (1-indexed), image: PIL.Image}
    """
    images = convert_from_path(str(pdf_path), dpi=150)
    return [
        {"page_number": i + 1, "image": img}
        for i, img in enumerate(images)
    ]


def load_pdf_from_bytes(data: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        return load_pdf_pages(tmp_path)
    finally:
        os.unlink(tmp_path)
