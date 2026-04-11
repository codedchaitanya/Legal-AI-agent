"""
Hybrid legal document summarisation: Legal-BERT (extractive) + PEGASUS (abstractive).

Integrated from AI_legal_agent. This is the LOCAL summarisation pipeline —
no API calls required. Used as primary summariser with Claude API as fallback.

Provides:
    load_summarization_models() → model bundle dict
    run_local_summarization(text, models) → {extractive_summary, entities, abstractive_summary}
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import nltk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # legal-ai-in-main/
_NLTK_DATA_DIR = _PROJECT_ROOT / "nltk_data"
_NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
if str(_NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_DATA_DIR))

logger = logging.getLogger(__name__)

# --- Configuration ---
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
PEGASUS_MODEL = "google/pegasus-xsum"
LEGAL_BERT_MAX_TOKENS = 512
CHUNK_TOKEN_BUDGET = 480
DEFAULT_TOP_N_SENTENCES = 7
TOP_N_PER_CHUNK = 5
EMBEDDING_BATCH_SIZE = 16
MIN_SENT_WORDS = 5
MIN_SENT_CHARS = 30

# IPC → BNS mapping for Indian legal entity extraction
IPC_TO_BNS = {
    "302": "BNS 103", "304": "BNS 105", "307": "BNS 109",
    "376": "BNS 63",  "420": "BNS 318", "379": "BNS 303",
    "498A": "BNS 85", "406": "BNS 316", "354": "BNS 74",
    "306": "BNS 108", "304A": "BNS 106", "323": "BNS 115",
    "363": "BNS 137", "364": "BNS 138", "365": "BNS 139",
    "375": "BNS 63",  "377": "BNS 377", "392": "BNS 309",
    "395": "BNS 310", "397": "BNS 312", "411": "BNS 317",
    "120B": "BNS 61", "34": "BNS 3(5)", "149": "BNS 190",
}


def _ensure_nltk_punkt() -> None:
    dl = str(_NLTK_DATA_DIR)
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=dl, quiet=True)


def _merge_numbered_fragments(sentences: list[str]) -> list[str]:
    if not sentences:
        return []
    merged: list[str] = []
    i = 0
    enum_only = re.compile(r"^(?:\(?[0-9]+\)?[.)]|[IVXLCivxlc]+[.)])\s*$")
    while i < len(sentences):
        cur = sentences[i].strip()
        if i + 1 < len(sentences) and enum_only.match(cur):
            merged.append(f"{cur} {sentences[i + 1].strip()}".strip())
            i += 2
        else:
            merged.append(cur)
            i += 1
    return merged


def _sentences_for_ranking(sentences: list[str]) -> list[str]:
    kept = [s for s in sentences if len(s.split()) >= MIN_SENT_WORDS and len(s.strip()) >= MIN_SENT_CHARS]
    return kept if kept else sentences


def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_summarization_models(
    device=None,
    legal_bert_name: str = LEGAL_BERT_MODEL,
    pegasus_name: str = PEGASUS_MODEL,
) -> dict[str, Any]:
    """Load Legal-BERT, PEGASUS, and optionally spaCy. Returns model bundle dict."""
    import torch
    from transformers import AutoModel, AutoTokenizer, PegasusForConditionalGeneration

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle: dict[str, Any] = {"device": device}

    logger.info("Loading Legal-BERT from %s …", legal_bert_name)
    bundle["legal_tokenizer"] = AutoTokenizer.from_pretrained(legal_bert_name)
    bundle["legal_model"] = AutoModel.from_pretrained(legal_bert_name)
    bundle["legal_model"].to(device).eval()

    logger.info("Loading PEGASUS from %s …", pegasus_name)
    bundle["pegasus_tokenizer"] = AutoTokenizer.from_pretrained(pegasus_name)
    bundle["pegasus_model"] = PegasusForConditionalGeneration.from_pretrained(pegasus_name)
    bundle["pegasus_model"].to(device).eval()

    try:
        import spacy
        bundle["spacy_nlp"] = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        logger.warning("spaCy not available; using regex-only entity extraction.")
        bundle["spacy_nlp"] = None

    return bundle


# ---------------------------------------------------------------------------
# Token counting & chunking
# ---------------------------------------------------------------------------

def _token_length(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))


def _chunk_text(text: str, tokenizer, max_tokens: int = CHUNK_TOKEN_BUDGET) -> list[str]:
    _ensure_nltk_punkt()
    sentences = nltk.sent_tokenize(preprocess_text(text))
    if not sentences:
        clean = preprocess_text(text)
        return [clean] if clean else []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_length(sent, tokenizer)
        if sent_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current).strip())
                current, current_tokens = [], 0
            chunks.append(sent[:max_tokens * 4])
            continue
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Embeddings & extractive summary
# ---------------------------------------------------------------------------

def _mean_pooling(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _get_embeddings(sentences, tokenizer, model, device, batch_size=EMBEDDING_BATCH_SIZE):
    import torch
    import numpy as np
    if not sentences:
        return np.zeros((0, model.config.hidden_size), dtype="float32")

    all_emb = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=LEGAL_BERT_MAX_TOKENS, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = _mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            all_emb.append(pooled.cpu())
    return torch.cat(all_emb, dim=0).numpy()


def _extractive_single_chunk(chunk_text, legal_tokenizer, legal_model, device, top_n, batch_size=EMBEDDING_BATCH_SIZE):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    _ensure_nltk_punkt()
    chunk_text = preprocess_text(chunk_text)
    raw = [s.strip() for s in nltk.sent_tokenize(chunk_text) if s.strip()]
    sents = _sentences_for_ranking(_merge_numbered_fragments(raw))
    if not sents:
        return ""
    if len(sents) <= top_n:
        return " ".join(sents)

    emb = _get_embeddings(sents, legal_tokenizer, legal_model, device, batch_size)
    doc_vec = emb.mean(axis=0, keepdims=True)
    sims = cosine_similarity(emb, doc_vec).ravel()
    top_idx = set(np.argsort(-sims)[:top_n].tolist())
    return " ".join(sents[i] for i in range(len(sents)) if i in top_idx)


def extractive_summary(text, legal_tokenizer, legal_model, device, top_n=DEFAULT_TOP_N_SENTENCES, top_n_per_chunk=TOP_N_PER_CHUNK):
    text = preprocess_text(text)
    if not text:
        return ""
    if _token_length(text, legal_tokenizer) <= CHUNK_TOKEN_BUDGET:
        return _extractive_single_chunk(text, legal_tokenizer, legal_model, device, top_n)

    chunks = _chunk_text(text, legal_tokenizer)
    parts = [_extractive_single_chunk(ch, legal_tokenizer, legal_model, device, top_n_per_chunk) for ch in chunks]
    merged = " ".join(p for p in parts if p).strip()
    if not merged:
        return ""
    if _token_length(merged, legal_tokenizer) <= CHUNK_TOKEN_BUDGET:
        return merged
    return _extractive_single_chunk(merged, legal_tokenizer, legal_model, device, top_n)


# ---------------------------------------------------------------------------
# Entity extraction (with Indian legal section support)
# ---------------------------------------------------------------------------

def _extract_legal_sections(text: str) -> list[tuple[str, str]]:
    text = preprocess_text(text)
    patterns = [
        r"\b(?:Section|Sec\.?)\s+[0-9]+[A-Z]?(?:\.[0-9]+)*(?:\s*\([a-z]\))?\b",
        r"\bArticle\s+[IVXLC0-9]+(?:\.[0-9]+)*\b",
        r"\b(?:Clause|Paragraph)\s+[0-9]+(?:\.[0-9]+)*\b",
        r"\bBNS\s+(?:Section\s+)?\d+[A-Z]?\b",
        r"\bBNSS\s+(?:Section\s+)?\d+[A-Z]?\b",
        r"\bBSA\s+(?:Section\s+)?\d+[A-Z]?\b",
        r"\bIPC\s+(?:Section\s+)?\d+[A-Z]?\b",
        r"\b(?:§|U\.S\.C\.|USC)\s*[§]?\s*[0-9]+(?:\.[0-9]+)*\b",
    ]
    found = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            found.append(("LEGAL_SECTION", m.group(0)))
    return found


def _enrich_with_bns(sections: list[str]) -> list[dict]:
    """Map old IPC sections to BNS equivalents where known."""
    enriched = []
    for s in sections:
        entry = {"raw": s, "bns_equivalent": ""}
        for ipc_num, bns_equiv in IPC_TO_BNS.items():
            if re.search(rf'\b{re.escape(ipc_num)}\b', s):
                entry["bns_equivalent"] = bns_equiv
                break
        enriched.append(entry)
    return enriched


def extract_entities(text: str, nlp=None) -> dict[str, Any]:
    """Extract PERSON, DATE, ORG, GPE, LEGAL_SECTION from text."""
    text = preprocess_text(text)
    out: dict[str, Any] = {"PERSON": [], "DATE": [], "ORG": [], "GPE": [], "LEGAL_SECTION": [], "sections_with_bns": []}

    sections_raw = []
    for _label, span in _extract_legal_sections(text):
        if span not in out["LEGAL_SECTION"]:
            out["LEGAL_SECTION"].append(span)
            sections_raw.append(span)
    out["sections_with_bns"] = _enrich_with_bns(sections_raw)

    if nlp is not None:
        try:
            doc = nlp(text[:1_000_000])
            for ent in doc.ents:
                if ent.label_ in out and ent.text.strip() and ent.text not in out[ent.label_]:
                    out[ent.label_].append(ent.text.strip())
        except Exception as e:
            logger.warning("Entity extraction failed: %s", e)

    return out


# ---------------------------------------------------------------------------
# Abstractive summary
# ---------------------------------------------------------------------------

def abstractive_summary(extractive_text, pegasus_tokenizer, pegasus_model, device, max_length=150, min_length=40):
    import torch
    extractive_text = preprocess_text(extractive_text)
    if not extractive_text:
        return ""
    inputs = pegasus_tokenizer(extractive_text, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        gen_ids = pegasus_model.generate(
            **inputs, max_length=max_length, min_length=min_length,
            num_beams=4, length_penalty=2.0, early_stopping=True,
        )
    return pegasus_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Full local pipeline
# ---------------------------------------------------------------------------

def run_local_summarization(
    text: str, models: dict[str, Any],
    top_n_extractive: int = DEFAULT_TOP_N_SENTENCES,
) -> dict[str, Any]:
    """
    End-to-end LOCAL summarisation: preprocess → extractive → entities → abstractive.
    No API calls — runs entirely on Legal-BERT + PEGASUS.
    """
    device = models["device"]
    clean = preprocess_text(text)
    if not clean:
        return {"extractive_summary": "", "entities": {}, "abstractive_summary": "", "sections_mentioned": []}

    ext = extractive_summary(clean, models["legal_tokenizer"], models["legal_model"], device, top_n=top_n_extractive)
    entities = extract_entities(clean, models.get("spacy_nlp"))

    try:
        abst = abstractive_summary(ext, models["pegasus_tokenizer"], models["pegasus_model"], device)
    except Exception:
        abst = ext

    return {
        "extractive_summary": ext,
        "entities": entities,
        "abstractive_summary": abst,
        "sections_mentioned": entities.get("sections_with_bns", []),
    }
