#!/usr/bin/env python3
"""
Indian Kanoon judgment scraper for QLoRA training data.

Indian Kanoon (https://indiankanoon.org) is the largest free Indian legal
search engine. This scraper fetches court judgments by domain-specific
keywords and saves them as structured JSONL for training.

Usage:
    # Scrape criminal_violent domain (50 judgments)
    python training/data_prep/kanoon_scraper.py \\
        --domain criminal_violent \\
        --max-results 50 \\
        --output data/raw/criminal_violent_judgments.jsonl

    # Scrape all domains (default 200 per domain)
    python training/data_prep/kanoon_scraper.py --all --max-results 200

HOW TO SCRAPE:
    1. Indian Kanoon provides search results at:
       https://indiankanoon.org/search/?formInput=<query>
    2. Individual judgments at: https://indiankanoon.org/doc/<doc_id>/
    3. The scraper searches by domain keywords, extracts judgment text,
       and formats into instruction-tuning pairs.
    4. Rate-limited to 1 request per 2 seconds to be respectful.
    5. Uses the public website — no API key needed.

IMPORTANT: Respect Indian Kanoon's terms of service. This scraper is for
educational/research purposes. For large-scale commercial use, contact
Indian Kanoon for their API access.
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

KANOON_SEARCH_URL = "https://indiankanoon.org/search/"
KANOON_DOC_URL = "https://indiankanoon.org/doc/{doc_id}/"
REQUEST_DELAY = 2.0  # seconds between requests
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) LegalAI-Research/1.0"

# Domain → search keywords (loaded from config if available, fallback here)
DOMAIN_KEYWORDS = {
    "criminal_violent": [
        "murder section 302", "culpable homicide 304", "attempt to murder 307",
        "grievous hurt section 325", "BNS 103 murder",
    ],
    "criminal_property": [
        "theft section 379", "robbery section 392", "cheating section 420",
        "criminal breach of trust 406", "dacoity section 395",
    ],
    "kidnapping_trafficking": [
        "kidnapping section 363", "abduction section 366",
        "human trafficking section 370", "wrongful confinement 342",
    ],
    "sexual_offences": [
        "rape section 376", "POCSO sexual assault", "stalking section 354D",
        "sexual harassment workplace", "molestation 354",
    ],
    "land_property": [
        "land dispute partition", "adverse possession property",
        "transfer of property act", "encroachment land revenue",
        "mutation revenue court",
    ],
    "family_matrimonial": [
        "divorce Hindu Marriage Act", "maintenance section 125",
        "child custody guardianship", "domestic violence protection",
        "498A cruelty husband",
    ],
    "constitutional": [
        "fundamental rights violation", "writ petition Article 226",
        "public interest litigation PIL", "habeas corpus Article 21",
        "right to equality Article 14",
    ],
    "corporate_commercial": [
        "company law winding up", "NCLT insolvency IBC",
        "SEBI securities regulation", "arbitration commercial dispute",
        "partnership dissolution",
    ],
    "labour_employment": [
        "industrial dispute retrenchment", "minimum wages violation",
        "provident fund EPF", "unfair termination labour court",
        "workmen compensation",
    ],
    "cyber_digital": [
        "IT Act cyber crime", "hacking section 66", "data theft digital",
        "electronic evidence 65B", "online fraud phishing",
    ],
    "tax_fiscal": [
        "income tax assessment appeal", "GST input tax credit",
        "customs duty evasion", "ITAT tribunal order",
        "tax penalty concealment",
    ],
    "civil_general": [
        "civil suit injunction", "specific performance contract",
        "consumer complaint deficiency", "money recovery suit",
        "declaratory suit title",
    ],
}


def _get_session():
    """Create a requests session with appropriate headers."""
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-IN,en;q=0.9",
    })
    return session


def search_kanoon(session, query: str, page: int = 0) -> list[dict]:
    """
    Search Indian Kanoon and extract doc IDs + titles from search results.
    Returns list of {doc_id, title, snippet}.
    """
    from bs4 import BeautifulSoup

    params = {"formInput": query, "pagenum": page}
    try:
        resp = session.get(KANOON_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Search failed for '%s': %s", query, e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for result_div in soup.select(".result"):
        title_elem = result_div.select_one(".result_title a")
        if not title_elem:
            continue
        href = title_elem.get("href", "")
        doc_id_match = re.search(r"/doc/(\d+)", href)
        if not doc_id_match:
            continue

        snippet_elem = result_div.select_one(".result_text")
        results.append({
            "doc_id": doc_id_match.group(1),
            "title": title_elem.get_text(strip=True),
            "snippet": snippet_elem.get_text(strip=True)[:500] if snippet_elem else "",
        })

    return results


def fetch_judgment(session, doc_id: str) -> dict | None:
    """
    Fetch full judgment text from Indian Kanoon.
    Returns {doc_id, title, court, date, text} or None.
    """
    from bs4 import BeautifulSoup

    url = KANOON_DOC_URL.format(doc_id=doc_id)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Failed to fetch doc %s: %s", doc_id, e)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    title_elem = soup.select_one("h2.doc_title")
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Extract court and date from doc_author/doc_bench
    court = ""
    date = ""
    author_elem = soup.select_one(".doc_author, .docsource_main")
    if author_elem:
        court = author_elem.get_text(strip=True)
    date_elem = soup.select_one(".doc_date, .doc_citations .citetxt")
    if date_elem:
        date = date_elem.get_text(strip=True)

    # Main judgment text
    judgment_div = soup.select_one("#judgments, .judgments, .expanded_text")
    if judgment_div:
        # Remove footnotes and headnotes
        for tag in judgment_div.select(".footnotes, .headnote_text"):
            tag.decompose()
        text = judgment_div.get_text(separator="\n", strip=True)
    else:
        # Fallback: grab all <p> inside the main content
        paragraphs = soup.select("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)

    if not text or len(text) < 200:
        logger.debug("Doc %s: text too short (%d chars), skipping.", doc_id, len(text))
        return None

    return {
        "doc_id": doc_id,
        "title": title,
        "court": court,
        "date": date,
        "text": text[:50000],  # cap at 50k chars per judgment
        "url": url,
    }


def scrape_domain(
    domain: str,
    max_results: int = 200,
    output_path: str | Path | None = None,
) -> list[dict]:
    """
    Scrape judgments for a single domain.
    Searches by each keyword, deduplicates by doc_id, fetches full text.
    """
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        logger.error("No keywords for domain '%s'.", domain)
        return []

    session = _get_session()
    seen_ids: set[str] = set()
    all_results: list[dict] = []
    per_keyword_limit = max(max_results // len(keywords), 10)

    logger.info("Scraping domain '%s' — %d keywords, target %d judgments.", domain, len(keywords), max_results)

    for keyword in keywords:
        if len(all_results) >= max_results:
            break

        logger.info("  Searching: '%s'", keyword)
        page = 0
        keyword_count = 0

        while keyword_count < per_keyword_limit and len(all_results) < max_results:
            time.sleep(REQUEST_DELAY)
            search_results = search_kanoon(session, keyword, page=page)
            if not search_results:
                break

            for sr in search_results:
                if sr["doc_id"] in seen_ids:
                    continue
                if len(all_results) >= max_results:
                    break

                time.sleep(REQUEST_DELAY)
                judgment = fetch_judgment(session, sr["doc_id"])
                if judgment:
                    judgment["domain"] = domain
                    judgment["search_keyword"] = keyword
                    all_results.append(judgment)
                    seen_ids.add(sr["doc_id"])
                    keyword_count += 1
                    logger.info("    [%d/%d] %s", len(all_results), max_results, judgment["title"][:80])

            page += 1
            if page > 5:  # don't go too deep in pagination
                break

    logger.info("Domain '%s': scraped %d judgments.", domain, len(all_results))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Saved to %s", output_path)

    return all_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scrape Indian Kanoon judgments for training data")
    parser.add_argument("--domain", type=str, help="Single domain to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all 12 domains")
    parser.add_argument("--max-results", type=int, default=200, help="Max judgments per domain")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (single domain)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output dir (--all mode)")
    args = parser.parse_args()

    if args.all:
        for domain in DOMAIN_KEYWORDS:
            out = Path(args.output_dir) / f"{domain}_judgments.jsonl"
            scrape_domain(domain, max_results=args.max_results, output_path=out)
    elif args.domain:
        out = args.output or f"data/raw/{args.domain}_judgments.jsonl"
        scrape_domain(args.domain, max_results=args.max_results, output_path=out)
    else:
        parser.error("Provide --domain <name> or --all")


if __name__ == "__main__":
    main()
