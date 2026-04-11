#!/usr/bin/env python3
# scripts/precompute_legal_db.py
"""
Run once to build and persist the Legal DB PageIndex.
Usage: python scripts/precompute_legal_db.py --input data/legal_db/bns_sections.json
"""
import asyncio
import argparse
from core.indexing.legal_db_precompute import precompute_from_file
from db.mongo import mongo
from db.redis_client import RedisClient


async def main(input_path: str):
    await mongo.connect()
    redis = RedisClient()

    print(f"Building legal DB index from {input_path}...")
    nodes = await precompute_from_file(input_path)

    print(f"Storing {len(nodes)} nodes in MongoDB...")
    for node in nodes:
        await mongo.legal_nodes.replace_one(
            {"node_id": node["node_id"]},
            node,
            upsert=True,
        )

    print("Warming Redis cache for legal nodes...")
    for node in nodes:
        await redis.set_legal_node(node["node_id"], node)

    print(f"Done. {len(nodes)} nodes indexed.")
    await mongo.disconnect()
    await redis.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.input))
