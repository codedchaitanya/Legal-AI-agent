# api/routes/documents.py
"""
Document upload endpoint — full pipeline:
  PDF/Image → OCR (GLM-OCR) → Local summarisation (Legal-BERT + PEGASUS)
  → Domain routing → PageIndex building → Store
"""
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from api.deps import get_redis, require_api_key
from db.redis_client import RedisClient
from db.mongo import mongo
from db.postgres import AsyncSessionLocal, Document

logger = logging.getLogger(__name__)
router = APIRouter()

_TEXT_EXTENSIONS = {".txt", ".text", ".md"}
_OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@router.post("/{case_id}/documents", status_code=201)
async def upload_document(
    case_id: str,
    file: UploadFile = File(...),
    redis: RedisClient = Depends(get_redis),
    _: str = Depends(require_api_key),
):
    doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
    raw_bytes = await file.read()
    filename = file.filename or "document.pdf"
    suffix = Path(filename).suffix.lower()

    # --- Step 1: Extract text ---
    text = ""
    was_ocr = False

    if suffix in _TEXT_EXTENSIONS:
        text = raw_bytes.decode("utf-8", errors="replace")
    elif suffix in _OCR_EXTENSIONS:
        # Use local OCR pipeline (GLM-OCR)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            from core.ingestion.ocr_module import load_ocr_model, ocr_file
            ocr_bundle = load_ocr_model()
            text = ocr_file(tmp_path, ocr_bundle)
            was_ocr = True
        except Exception as e:
            logger.warning("OCR failed, falling back to pypdf text extraction: %s", e)
            try:
                from pypdf import PdfReader
                reader = PdfReader(tmp_path)
                text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception:
                pass
        finally:
            os.unlink(tmp_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from document.")

    # --- Step 2: Summarise (local first, Claude fallback) ---
    from core.ingestion.summarizer import summarize_document
    from core.ingestion.doc_classifier import normalise_doc_type

    summary_result = await summarize_document(text, ocr_input=was_ocr)
    doc_type = normalise_doc_type(summary_result.get("doc_type", "other"))

    # --- Step 3: Build PageIndex ---
    from core.indexing.pageindex_builder import build_case_pageindex

    # Build page dicts for the index
    pages = [{"page_number": 1, "text": text}]
    if was_ocr and suffix == ".pdf":
        # Split into per-page text approximation
        from core.ingestion.pdf_loader import load_pdf_from_bytes
        try:
            pdf_pages = load_pdf_from_bytes(raw_bytes)
            pages = [{"page_number": p["page_number"], "text": ""} for p in pdf_pages]
        except Exception:
            pass

    await redis.invalidate_case_pageindex(case_id)
    tree = await build_case_pageindex(case_id, doc_id, doc_type, pages)
    await mongo.case_indexes.replace_one(
        {"case_id": case_id, "doc_id": doc_id}, tree, upsert=True
    )
    await redis.set_case_pageindex(case_id, tree)

    # --- Step 4: Store metadata ---
    async with AsyncSessionLocal() as db:
        db.add(Document(
            id=doc_id, case_id=case_id, doc_type=doc_type,
            filename=filename, page_count=len(pages),
            storage_path=f"{case_id}/{doc_id}/{filename}",
        ))
        await db.commit()

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "pages": len(pages),
        "summary": summary_result.get("summary", ""),
        "sections_mentioned": summary_result.get("sections_mentioned", []),
        "summary_source": summary_result.get("source", "unknown"),
    }
