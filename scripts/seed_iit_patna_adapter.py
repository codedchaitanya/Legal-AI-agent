#!/usr/bin/env python3
# scripts/seed_iit_patna_adapter.py
"""
Download IIT Patna Indian legal QLoRA adapter as criminal_v0 baseline.
Usage: python scripts/seed_iit_patna_adapter.py
"""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="RMani1/indian-legal-qwen-lora",
    local_dir="adapters/criminal"
)
print("IIT Patna criminal_v0 adapter downloaded to adapters/criminal")
