#!/usr/bin/env python3
"""
Split a combined training JSONL into per-domain files + train/eval splits.

Usage:
    # Split a combined file by domain
    python training/data_prep/domain_splitter.py \\
        --input data/training/all_pairs.jsonl \\
        --output-dir data/training/ \\
        --eval-ratio 0.1

    # Merge multiple domain files and re-split
    python training/data_prep/domain_splitter.py \\
        --input-dir data/training/raw/ \\
        --output-dir data/training/ \\
        --eval-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

DOMAINS = [
    "criminal_violent", "criminal_property", "kidnapping_trafficking",
    "sexual_offences", "land_property", "family_matrimonial",
    "constitutional", "corporate_commercial", "labour_employment",
    "cyber_digital", "tax_fiscal", "civil_general",
]


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_by_domain(items: list[dict]) -> dict[str, list[dict]]:
    """Group items by their 'domain' field."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        domain = item.get("domain", "civil_general")
        if domain not in DOMAINS:
            domain = "civil_general"
        groups[domain].append(item)
    return groups


def train_eval_split(items: list[dict], eval_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split items into train and eval sets."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Split training data by domain with train/eval splits")
    parser.add_argument("--input", type=str, help="Single JSONL file to split")
    parser.add_argument("--input-dir", type=str, help="Directory of JSONL files to merge and split")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction for eval split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_items = []
    if args.input:
        all_items = load_jsonl(Path(args.input))
    elif args.input_dir:
        for jsonl_file in Path(args.input_dir).glob("*.jsonl"):
            all_items.extend(load_jsonl(jsonl_file))
    else:
        parser.error("Provide --input or --input-dir")

    logger.info("Loaded %d total items.", len(all_items))

    domain_groups = split_by_domain(all_items)
    output_dir = Path(args.output_dir)

    for domain in DOMAINS:
        items = domain_groups.get(domain, [])
        if not items:
            logger.warning("Domain '%s': no data.", domain)
            continue

        train, eval_set = train_eval_split(items, eval_ratio=args.eval_ratio, seed=args.seed)

        train_path = output_dir / f"{domain}_train.jsonl"
        eval_path = output_dir / f"{domain}_eval.jsonl"
        save_jsonl(train, train_path)
        save_jsonl(eval_set, eval_path)

        logger.info("  %s: %d train, %d eval", domain, len(train), len(eval_set))

    logger.info("Done. Files saved to %s", output_dir)


if __name__ == "__main__":
    main()
