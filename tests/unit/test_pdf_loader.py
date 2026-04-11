# tests/unit/test_pdf_loader.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_pdf_to_pages_returns_list_of_page_dicts():
    from core.ingestion.pdf_loader import load_pdf_pages
    with patch("core.ingestion.pdf_loader.convert_from_path") as mock_convert:
        mock_img = MagicMock()
        mock_convert.return_value = [mock_img, mock_img]
        pages = load_pdf_pages("fake.pdf")
        assert len(pages) == 2
        assert "page_number" in pages[0]
        assert "image" in pages[0]


def test_pdf_page_numbers_are_one_indexed():
    from core.ingestion.pdf_loader import load_pdf_pages
    with patch("core.ingestion.pdf_loader.convert_from_path") as mock_convert:
        mock_img = MagicMock()
        mock_convert.return_value = [mock_img]
        pages = load_pdf_pages("fake.pdf")
        assert pages[0]["page_number"] == 1
