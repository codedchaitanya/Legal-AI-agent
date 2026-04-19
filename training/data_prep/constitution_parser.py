#!/usr/bin/env python3
"""
Parse the Indian Constitution text file into structured JSON and training data.

Reads the raw Constitution text, extracts each Article with its number, title,
full text, Part, and maps it to the relevant legal training domain(s).

Usage:
    python training/data_prep/constitution_parser.py \
        --input data/constitution_of_India \
        --output-json data/legal_db/constitution_articles.json \
        --output-qa data/training/constitution_articles_qa.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── Article-range to domain mapping ──────────────────────────────────────────
# An article can belong to multiple domains if relevant.
ARTICLE_DOMAIN_MAP = {
    # Part I: Union & Territory (1-4)
    range(1, 5): ["constitutional"],
    # Part II: Citizenship (5-11)
    range(5, 12): ["constitutional"],
    # Part III: Fundamental Rights (12-35)
    range(12, 36): ["constitutional", "criminal_violent"],
    # Part IV: Directive Principles (36-51)
    range(36, 52): ["constitutional", "labour_employment"],
    # Part V: The Union — Executive (52-78)
    range(52, 79): ["constitutional"],
    # Part V: Parliament (79-122)
    range(79, 123): ["constitutional"],
    # Part V: Legislative powers (123-151)
    range(123, 152): ["constitutional"],
    # Part VI: The States (152-237)
    range(152, 238): ["constitutional"],
    # Part VIII: Union Territories (239-242)
    range(239, 243): ["constitutional"],
    # Part IX: Panchayats (243-243O → use 243-244)
    range(243, 244): ["constitutional"],
    # Part X: Scheduled/Tribal Areas (244-244A)
    range(244, 245): ["constitutional"],
    # Part XI: Union-State Relations (245-263)
    range(245, 264): ["constitutional", "tax_fiscal"],
    # Part XII: Finance, Property, Contracts (264-300A)
    range(264, 301): ["constitutional", "tax_fiscal", "land_property"],
    # Part XIII: Trade & Commerce (301-307)
    range(301, 308): ["constitutional", "corporate_commercial"],
    # Part XIV: Services (308-323)
    range(308, 324): ["constitutional", "labour_employment"],
    # Part XV: Elections (324-329)
    range(324, 330): ["constitutional"],
    # Part XVI: Special Provisions for certain classes (330-342)
    range(330, 343): ["constitutional"],
    # Part XVII: Official Languages (343-351)
    range(343, 352): ["constitutional"],
    # Part XVIII: Emergency Provisions (352-360)
    range(352, 361): ["constitutional"],
    # Part XIX: Miscellaneous (361-367)
    range(361, 368): ["constitutional"],
    # Part XX: Amendment (368)
    range(368, 369): ["constitutional"],
    # Part XXI: Temporary/Transitional (369-392)
    range(369, 393): ["constitutional"],
    # Part XXII: Short title, Commencement (393-395)
    range(393, 396): ["constitutional"],
}

# Part name lookup by article range
ARTICLE_PART_MAP = {
    range(1, 5): "Part I — Union and its Territory",
    range(5, 12): "Part II — Citizenship",
    range(12, 36): "Part III — Fundamental Rights",
    range(36, 52): "Part IV — Directive Principles of State Policy",
    range(52, 152): "Part V — The Union",
    range(152, 238): "Part VI — The States",
    range(239, 243): "Part VIII — The Union Territories",
    range(243, 244): "Part IX — The Panchayats",
    range(244, 245): "Part X — Scheduled and Tribal Areas",
    range(245, 264): "Part XI — Relations between the Union and the States",
    range(264, 301): "Part XII — Finance, Property, Contracts and Suits",
    range(301, 308): "Part XIII — Trade, Commerce and Intercourse",
    range(308, 324): "Part XIV — Services under the Union and States",
    range(324, 330): "Part XV — Elections",
    range(330, 343): "Part XVI — Special Provisions Relating to Certain Classes",
    range(343, 352): "Part XVII — Official Languages",
    range(352, 361): "Part XVIII — Emergency Provisions",
    range(361, 368): "Part XIX — Miscellaneous",
    range(368, 369): "Part XX — Amendment of the Constitution",
    range(369, 393): "Part XXI — Temporary, Transitional and Special Provisions",
    range(393, 396): "Part XXII — Short Title, Commencement and Repeals",
}


def _get_domains_for_article(article_num: int) -> list[str]:
    for r, domains in ARTICLE_DOMAIN_MAP.items():
        if article_num in r:
            return domains
    return ["constitutional"]


def _get_part_for_article(article_num: int) -> str:
    for r, part in ARTICLE_PART_MAP.items():
        if article_num in r:
            return part
    return "Unknown"


def parse_constitution(input_path: str | Path) -> list[dict]:
    """Parse the Constitution text file into a list of article dicts."""
    input_path = Path(input_path)
    text = input_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Skip the CSV header (Parts table) — articles start after "Articles" line
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "Articles":
            start_idx = i + 1
            break

    article_lines = lines[start_idx:]
    raw_text = "\n".join(article_lines)

    # Match article numbers: handles "1.", "2A.", "51A.", "243.", "243A.", "371A.", etc.
    # Each article starts with the number followed by a period and space/title
    article_pattern = re.compile(
        r'^"?(\d+[A-Z]?)\.\s+(.+?)(?:\n|$)',
        re.MULTILINE,
    )

    matches = list(article_pattern.finditer(raw_text))
    articles = []

    for i, match in enumerate(matches):
        article_id = match.group(1)
        # Get the full text until the next article
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        full_text = raw_text[start:end].strip()

        # Clean up: remove surrounding quotes
        full_text = full_text.strip('"').strip()

        # Extract the title (first sentence/phrase before the substantive text)
        title_match = re.match(r'(\d+[A-Z]?)\.\s+(.+?)(?:\s*\(1\)|\s+[A-Z][a-z])', full_text)
        if title_match:
            title = title_match.group(2).strip()
        else:
            first_line = full_text.split("\n")[0]
            title = re.sub(r'^\d+[A-Z]?\.\s*', '', first_line).strip()

        # Truncate overly long titles
        if len(title) > 200:
            title = title[:200].rsplit(" ", 1)[0] + "…"

        # Get the numeric part for domain mapping
        num_match = re.match(r'(\d+)', article_id)
        article_num = int(num_match.group(1)) if num_match else 0

        domains = _get_domains_for_article(article_num)
        part = _get_part_for_article(article_num)

        articles.append({
            "article_id": article_id,
            "article_number": article_num,
            "title": title,
            "text": full_text,
            "part": part,
            "domains": domains,
            "source": "Constitution of India",
        })

    logger.info("Parsed %d articles from Constitution.", len(articles))
    return articles


# ── QA pair generation ───────────────────────────────────────────────────────

CONSTITUTION_QA_TEMPLATES = [
    "Explain Article {article_id} of the Indian Constitution. What does it provide for?",
    "What are the key provisions of Article {article_id} of the Constitution of India?",
    "What rights or obligations does Article {article_id} establish under the Indian Constitution?",
    "Describe the scope and significance of Article {article_id} ({part}).",
    "How is Article {article_id} of the Indian Constitution applied in practice?",
]

CHAT_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""


def _load_system_prompt(domain: str) -> str:
    config_path = Path(__file__).parent.parent / "configs" / f"{domain}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("system_prompt", "").strip()
    return "You are an expert Indian legal assistant. Answer precisely, citing relevant constitutional provisions."


def generate_qa_pairs(articles: list[dict], pairs_per_article: int = 2) -> list[dict]:
    """Generate training QA pairs from Constitution articles."""
    all_pairs = []

    for article in articles:
        if len(article["text"]) < 50:
            continue

        for domain in article["domains"]:
            system_prompt = _load_system_prompt(domain)
            templates = random.sample(CONSTITUTION_QA_TEMPLATES, min(pairs_per_article, len(CONSTITUTION_QA_TEMPLATES)))

            for idx, template in enumerate(templates):
                question = template.format(
                    article_id=article["article_id"],
                    part=article["part"],
                )

                answer = f"Article {article['article_id']} of the Constitution of India ({article['part']}):\n\n{article['text']}"
                if len(answer) > 3000:
                    answer = answer[:3000]

                formatted = CHAT_TEMPLATE.format(
                    system_prompt=system_prompt,
                    question=question,
                    answer=answer.strip(),
                )

                all_pairs.append({
                    "text": formatted,
                    "domain": domain,
                    "source": "constitution",
                    "article_id": article["article_id"],
                    "pair_idx": idx,
                })

    random.shuffle(all_pairs)
    logger.info("Generated %d QA pairs from %d articles.", len(all_pairs), len(articles))
    return all_pairs


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Parse Indian Constitution into structured data and training pairs")
    parser.add_argument("--input", required=True, help="Path to Constitution text file")
    parser.add_argument("--output-json", default="data/legal_db/constitution_articles.json", help="Output JSON path")
    parser.add_argument("--output-qa", default="data/training/constitution_articles_qa.jsonl", help="Output QA JSONL")
    parser.add_argument("--pairs-per-article", type=int, default=2, help="QA pairs per article per domain")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Parse
    articles = parse_constitution(args.input)

    # Save structured JSON
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d articles to %s", len(articles), json_path)

    # Generate QA pairs
    pairs = generate_qa_pairs(articles, pairs_per_article=args.pairs_per_article)

    # Save QA JSONL
    qa_path = Path(args.output_qa)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("Saved %d QA pairs to %s", len(pairs), qa_path)

    # Summary by domain
    domain_counts: dict[str, int] = {}
    for p in pairs:
        domain_counts[p["domain"]] = domain_counts.get(p["domain"], 0) + 1
    logger.info("QA pairs by domain:")
    for d, c in sorted(domain_counts.items()):
        logger.info("  %s: %d", d, c)


if __name__ == "__main__":
    main()
