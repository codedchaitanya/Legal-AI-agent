#!/usr/bin/env python3
# tests/eval/run_inlegalbench.py
"""
Run InLegalBench evaluation against current pipeline.
Usage: python tests/eval/run_inlegalbench.py --dataset data/eval/inlegalbench_sample.json

Expected output: accuracy score + per-category breakdown.
This baseline MUST be recorded before any LoRA adapter training begins.
It establishes whether LoRA training is worth the 6-week data prep investment.

Decision rule:
  - If accuracy >= 65% -> LoRA is optional enhancement
  - If accuracy < 55% -> LoRA is necessary, proceed with adapter training
"""
import asyncio
import json
import argparse
from pathlib import Path


async def evaluate(dataset_path: str) -> dict:
    from core.routing.domain_router import classify_domains
    from core.routing.adapter_selector import select_adapters
    from core.reasoning.case_research import run_case_research
    from db.redis_client import RedisClient

    dataset = json.loads(Path(dataset_path).read_text())
    redis = RedisClient()
    correct = 0
    results_by_domain: dict[str, dict] = {}

    for i, item in enumerate(dataset):
        query = item["question"]
        expected = item["answer"].strip().lower()
        domain = item.get("domain", "unknown")

        domain_scores = await classify_domains(query, redis=redis)
        adapters = select_adapters(domain_scores)

        result = await run_case_research(
            query=query,
            case_id="EVAL-CASE",
            adapters=adapters,
            session_history=[],
            redis=redis,
        )
        predicted = result["answer"].strip().lower()
        is_correct = expected in predicted or predicted in expected

        if is_correct:
            correct += 1
        results_by_domain.setdefault(domain, {"correct": 0, "total": 0})
        results_by_domain[domain]["total"] += 1
        if is_correct:
            results_by_domain[domain]["correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(dataset)}, running accuracy: {correct/(i+1):.2%}")

    accuracy = correct / len(dataset) if dataset else 0.0
    print(f"\n=== InLegalBench Results ===")
    print(f"Overall accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")
    print("\nPer-domain breakdown:")
    for domain, stats in results_by_domain.items():
        acc = stats["correct"] / stats["total"]
        print(f"  {domain}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    await redis.close()
    return {"accuracy": accuracy, "by_domain": results_by_domain}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to InLegalBench JSON file")
    args = parser.parse_args()
    asyncio.run(evaluate(args.dataset))
