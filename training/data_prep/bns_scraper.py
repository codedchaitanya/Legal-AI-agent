#!/usr/bin/env python3
"""
BNS / BNSS / BSA section scraper and formatter.

Downloads the full text of:
  - Bharatiya Nyaya Sanhita (BNS) 2023 — replaces IPC
  - Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 — replaces CrPC
  - Bharatiya Sakshya Adhiniyam (BSA) 2023 — replaces Indian Evidence Act

Sources:
  - India Code: https://www.indiacode.nic.in/
  - Legislative Dept: https://legislative.gov.in/
  - eGazette: https://egazette.gov.in/

The script scrapes section-by-section text and saves structured JSON
suitable for building the Legal DB PageIndex.

Usage:
    # Scrape BNS sections
    python training/data_prep/bns_scraper.py --act bns --output data/legal_db/bns_sections.json

    # Scrape all three acts
    python training/data_prep/bns_scraper.py --all --output-dir data/legal_db/

HOW TO USE (if scraping is blocked):
    The script also provides a manual fallback mode where you can download
    the PDF of the act from indiacode.nic.in and extract sections locally.
    See the --from-pdf flag.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REQUEST_DELAY = 2.0
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) LegalAI-Research/1.0"

# India Code base URLs for each act
ACT_URLS = {
    "bns": {
        "name": "Bharatiya Nyaya Sanhita, 2023",
        "short": "BNS",
        "india_code_id": "45 of 2023",
        "search_url": "https://www.indiacode.nic.in/handle/123456789/19601",
        "total_sections": 358,
    },
    "bnss": {
        "name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "short": "BNSS",
        "india_code_id": "46 of 2023",
        "search_url": "https://www.indiacode.nic.in/handle/123456789/19602",
        "total_sections": 531,
    },
    "bsa": {
        "name": "Bharatiya Sakshya Adhiniyam, 2023",
        "short": "BSA",
        "india_code_id": "47 of 2023",
        "search_url": "https://www.indiacode.nic.in/handle/123456789/19603",
        "total_sections": 170,
    },
}

# BNS section → domain mapping for the 12 adapters
BNS_DOMAIN_MAP = {
    range(1, 24): "civil_general",         # General exceptions, definitions
    range(24, 62): "civil_general",         # General provisions
    range(63, 100): "sexual_offences",      # Sexual offences
    range(100, 119): "criminal_violent",    # Offences against body
    range(119, 137): "civil_general",       # Criminal force, assault
    range(137, 145): "kidnapping_trafficking",
    range(145, 190): "criminal_violent",    # Wrongful restraint etc.
    range(190, 250): "constitutional",      # Offences against state
    range(250, 303): "civil_general",       # Public tranquility, etc.
    range(303, 334): "criminal_property",   # Offences against property
    range(334, 358): "civil_general",       # Miscellaneous
}


def _get_bns_domain(section_num: int) -> str:
    """Map a BNS section number to one of the 12 domains."""
    for sec_range, domain in BNS_DOMAIN_MAP.items():
        if section_num in sec_range:
            return domain
    return "civil_general"


# IPC ↔ BNS equivalence table (common sections)
IPC_BNS_EQUIVALENCE = {
    "302": 103, "304": 105, "304A": 106, "306": 108, "307": 109,
    "323": 115, "325": 117, "326": 118, "354": 74, "354A": 75,
    "354D": 78, "363": 137, "364": 138, "365": 139, "375": 63,
    "376": 64, "376A": 66, "377": 377, "379": 303, "380": 305,
    "392": 309, "395": 310, "397": 312, "406": 316, "420": 318,
    "498A": 85, "499": 356, "500": 357, "34": "3(5)", "120B": 61,
    "149": 190, "302/34": "103/3(5)",
}


def scrape_india_code(act_key: str) -> list[dict]:
    """
    Scrape sections from India Code website.
    Returns list of section dicts.

    NOTE: India Code may block automated access. If this fails,
    use --from-pdf mode instead.
    """
    import requests
    from bs4 import BeautifulSoup

    act_info = ACT_URLS[act_key]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    logger.info("Scraping %s from India Code …", act_info["name"])

    sections = []
    try:
        resp = session.get(act_info["search_url"], timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # India Code stores sections as links within the act page
        section_links = soup.select("a[href*='/bitstream/']")
        logger.info("Found %d section links.", len(section_links))

        for link in section_links:
            time.sleep(REQUEST_DELAY)
            href = link.get("href", "")
            title = link.get_text(strip=True)

            sec_match = re.search(r'[Ss]ection\s+(\d+[A-Z]?)', title)
            sec_num = sec_match.group(1) if sec_match else ""

            try:
                sec_resp = session.get(f"https://www.indiacode.nic.in{href}", timeout=30)
                sec_soup = BeautifulSoup(sec_resp.text, "html.parser")
                content = sec_soup.select_one(".section-content, .act-content, main")
                text = content.get_text(separator="\n", strip=True) if content else ""
            except Exception as e:
                logger.warning("Failed to fetch section %s: %s", sec_num, e)
                text = ""

            if text:
                domain = _get_bns_domain(int(sec_num)) if sec_num.isdigit() else "civil_general"
                old_equiv = ""
                for ipc_sec, bns_sec in IPC_BNS_EQUIVALENCE.items():
                    if str(bns_sec) == sec_num:
                        old_equiv = f"IPC Section {ipc_sec}"
                        break

                sections.append({
                    "act": act_info["name"],
                    "act_short": act_info["short"],
                    "section_number": sec_num,
                    "title": title,
                    "text": text[:10000],
                    "domain": domain,
                    "old_equivalent": old_equiv,
                    "keywords": _extract_keywords(text),
                })

    except Exception as e:
        logger.error("Scraping failed: %s", e)
        logger.info("TIP: Use --from-pdf mode if India Code blocks requests.")

    return sections


def extract_sections_from_pdf(pdf_path: str | Path, act_key: str) -> list[dict]:
    """
    Extract sections from a downloaded PDF of the act.
    Download PDFs from: https://www.indiacode.nic.in/ or https://legislative.gov.in/

    For BNS: https://www.indiacode.nic.in/bitstream/123456789/19601/1/BNS.pdf
    For BNSS: search on indiacode.nic.in
    For BSA: search on indiacode.nic.in
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pip install pypdf")

    act_info = ACT_URLS[act_key]
    reader = PdfReader(str(pdf_path))
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    # Split on "Section N." or "N." at the start of lines
    section_pattern = re.compile(
        r'(?:^|\n)\s*(?:Section\s+)?(\d+[A-Z]?)\.\s*[-—]?\s*(.+?)(?=\n\s*(?:Section\s+)?\d+[A-Z]?\.\s|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    sections = []
    for match in section_pattern.finditer(full_text):
        sec_num = match.group(1)
        text = match.group(2).strip()
        title_line = text.split("\n")[0][:200]

        domain = _get_bns_domain(int(sec_num)) if sec_num.isdigit() else "civil_general"
        old_equiv = ""
        for ipc_sec, bns_sec in IPC_BNS_EQUIVALENCE.items():
            if str(bns_sec) == sec_num:
                old_equiv = f"IPC Section {ipc_sec}"
                break

        sections.append({
            "act": act_info["name"],
            "act_short": act_info["short"],
            "section_number": sec_num,
            "title": f"Section {sec_num} — {title_line}",
            "text": text[:10000],
            "domain": domain,
            "old_equivalent": old_equiv,
            "keywords": _extract_keywords(text),
        })

    logger.info("Extracted %d sections from PDF.", len(sections))
    return sections


def _extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract top legal keywords from section text."""
    legal_terms = [
        "punishment", "imprisonment", "fine", "death", "rigorous", "simple",
        "cognizable", "non-cognizable", "bailable", "non-bailable",
        "compoundable", "warrant", "summons", "complaint", "FIR",
        "accused", "victim", "complainant", "witness", "evidence",
        "offence", "crime", "penalty", "sentence", "conviction",
        "acquittal", "bail", "anticipatory bail", "appeal", "revision",
        "murder", "theft", "cheating", "fraud", "assault", "kidnapping",
        "rape", "robbery", "extortion", "forgery", "defamation",
        "conspiracy", "abetment", "attempt", "negligence",
    ]
    text_lower = text.lower()
    found = [t for t in legal_terms if t in text_lower]
    return found[:max_keywords]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scrape BNS/BNSS/BSA section text")
    parser.add_argument("--act", choices=["bns", "bnss", "bsa"], help="Which act to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all three acts")
    parser.add_argument("--from-pdf", type=str, help="Extract from downloaded PDF instead of scraping")
    parser.add_argument("--output", type=str, help="Output JSON path (single act)")
    parser.add_argument("--output-dir", type=str, default="data/legal_db", help="Output dir (--all mode)")
    args = parser.parse_args()

    acts = list(ACT_URLS.keys()) if args.all else ([args.act] if args.act else [])
    if not acts:
        parser.error("Provide --act <bns|bnss|bsa> or --all")

    for act_key in acts:
        if args.from_pdf:
            sections = extract_sections_from_pdf(args.from_pdf, act_key)
        else:
            sections = scrape_india_code(act_key)

        out_path = args.output or str(Path(args.output_dir) / f"{act_key}_sections.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d sections to %s", len(sections), out_path)


if __name__ == "__main__":
    main()
