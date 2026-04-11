"""
OCR front-end using GLM-OCR (zai-org/GLM-OCR).

Integrated from the AI_legal_agent pipeline. Handles:
  1. Load GLM-OCR model and processor from HuggingFace
  2. Convert PDF pages → PIL Images (PyMuPDF primary, pdf2image fallback)
  3. Run OCR over each image and concatenate page text
  4. Post-OCR text cleaning (noise, whitespace, hyphenation)

Supported input formats: .pdf  .png  .jpg  .jpeg  .tiff  .tif  .bmp  .webp
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GLM_OCR_MODEL_ID = "zai-org/GLM-OCR"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSION = ".pdf"

OCR_PROMPT = (
    "Please extract and transcribe all text from this image exactly as it appears, "
    "preserving paragraph breaks. Do not summarize or interpret the content."
)
OCR_MAX_NEW_TOKENS = 2048
PDF_RENDER_DPI = 200


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ocr_model(
    model_id: str = GLM_OCR_MODEL_ID,
    device: str | None = None,
) -> dict[str, Any]:
    """Load GLM-OCR processor and model. Returns dict with processor, model, device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading GLM-OCR processor from %s …", model_id)
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load GLM-OCR processor: {exc}") from exc

    logger.info("Loading GLM-OCR model from %s onto %s …", model_id, device)
    try:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype, device_map=device,
        )
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to load GLM-OCR model: {exc}") from exc

    logger.info("GLM-OCR loaded successfully.")
    return {"processor": processor, "model": model, "device": device}


# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------

def pdf_to_images(file_path: str | Path, dpi: int = PDF_RENDER_DPI) -> list[Any]:
    """Rasterise PDF pages into PIL Images. Tries PyMuPDF, falls back to pdf2image."""
    path = Path(file_path)
    logger.info("Converting PDF '%s' to images at %d DPI …", path.name, dpi)

    try:
        import fitz  # PyMuPDF
        from PIL import Image

        doc = fitz.open(str(path))
        images: list[Any] = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        doc.close()
        logger.info("PDF → %d page image(s) via PyMuPDF.", len(images))
        return images
    except ImportError:
        logger.debug("PyMuPDF not available; trying pdf2image.")
    except Exception as exc:
        logger.warning("PyMuPDF failed (%s); trying pdf2image.", exc)

    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), dpi=dpi)
        logger.info("PDF → %d page image(s) via pdf2image.", len(images))
        return images
    except ImportError as exc:
        raise ImportError(
            "PDF rasterisation requires PyMuPDF or pdf2image.\n"
            "  pip install pymupdf       (recommended)\n"
            "  pip install pdf2image     (also needs Poppler)"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not convert PDF '{path}' to images: {exc}") from exc


def load_input_file(file_path: str | Path) -> tuple[str, list[Any]]:
    """Return (file_type, images) for a PDF or image file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == PDF_EXTENSION:
        return "pdf", pdf_to_images(path)
    if suffix in IMAGE_EXTENSIONS:
        from PIL import Image
        return "image", [Image.open(path).convert("RGB")]

    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        f"Supported: {PDF_EXTENSION} and {', '.join(sorted(IMAGE_EXTENSIONS))}"
    )


# ---------------------------------------------------------------------------
# OCR inference
# ---------------------------------------------------------------------------

def _ocr_single_image(
    image: Any, processor: Any, model: Any, device: str,
    max_new_tokens: int = OCR_MAX_NEW_TOKENS,
) -> str:
    """Run GLM-OCR on one PIL Image and return extracted text."""
    import torch

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": OCR_PROMPT},
        ]}
    ]
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][prompt_len:]
        return processor.decode(generated, skip_special_tokens=True).strip()
    except Exception as exc:
        logger.error("OCR inference failed: %s", exc)
        return ""


def perform_ocr(
    images: list[Any], ocr_bundle: dict[str, Any],
    max_new_tokens: int = OCR_MAX_NEW_TOKENS,
) -> str:
    """Run GLM-OCR over a list of PIL images. Returns concatenated page text."""
    if not images:
        return ""

    processor, model, device = ocr_bundle["processor"], ocr_bundle["model"], ocr_bundle["device"]
    page_texts: list[str] = []

    for idx, img in enumerate(images):
        logger.info("OCR: processing page %d / %d …", idx + 1, len(images))
        text = _ocr_single_image(img, processor, model, device, max_new_tokens)
        if text:
            page_texts.append(text)

    full_text = "\n\n".join(page_texts)
    logger.info("OCR complete: %d page(s), %d total characters.", len(images), len(full_text))
    return full_text


# ---------------------------------------------------------------------------
# Text cleaning (post-OCR)
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """Post-OCR cleaning: line endings, soft hyphens, noise, whitespace."""
    if not text or not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"-\n(\S)", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"(?m)^[|~`\^]{1,3}\s*$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# High-level: file → cleaned text
# ---------------------------------------------------------------------------

def ocr_file(
    file_path: str | Path, ocr_bundle: dict[str, Any],
    dpi: int = PDF_RENDER_DPI, max_new_tokens: int = OCR_MAX_NEW_TOKENS,
) -> str:
    """file_path → cleaned OCR text string."""
    file_type, images = load_input_file(file_path)
    logger.info("File type: %s | Pages/images: %d", file_type.upper(), len(images))
    raw_text = perform_ocr(images, ocr_bundle, max_new_tokens=max_new_tokens)
    return clean_ocr_text(raw_text)
