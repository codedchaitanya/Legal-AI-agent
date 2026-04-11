#!/usr/bin/env python3
"""
QLoRA adapter training script for Indian legal domain adapters.

Trains a single LoRA adapter on top of Qwen2.5-7B-Instruct using domain-specific
legal data. Designed to run on Google Colab (T4 or A100 GPU).

Usage:
    # Train criminal_violent adapter
    python training/train_adapter.py --config training/configs/criminal_violent.yaml

    # Train with custom base config overrides
    python training/train_adapter.py \\
        --config training/configs/criminal_violent.yaml \\
        --epochs 5 \\
        --learning-rate 1e-4 \\
        --output-dir adapters/criminal_violent

    # Resume from checkpoint
    python training/train_adapter.py \\
        --config training/configs/criminal_violent.yaml \\
        --resume adapters/criminal_violent/checkpoint-500

For Colab: Copy this script + configs + data to your Drive, then run.
See training/colab_train.py for a Colab-optimised wrapper.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: dict | None = None) -> dict:
    """Load training config, merging base + domain-specific + CLI overrides."""
    config_path = Path(config_path)
    with open(config_path) as f:
        domain_config = yaml.safe_load(f)

    # Load base config
    base_ref = domain_config.pop("_base", "base.yaml")
    base_path = config_path.parent / base_ref
    if base_path.exists():
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {}

    # Deep merge: base → domain → overrides
    config = _deep_merge(base_config, domain_config)
    if overrides:
        config = _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_dataset(train_file: str, eval_file: str | None = None, max_samples: int | None = None):
    """Load training data from JSONL files."""
    from datasets import Dataset

    def _load_jsonl(path):
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    train_items = _load_jsonl(train_file)
    if max_samples:
        train_items = train_items[:max_samples]

    train_ds = Dataset.from_list(train_items)

    eval_ds = None
    if eval_file and Path(eval_file).exists():
        eval_items = _load_jsonl(eval_file)
        eval_ds = Dataset.from_list(eval_items)

    return train_ds, eval_ds


def train(config: dict, output_dir: str | None = None, resume_from: str | None = None):
    """Run QLoRA fine-tuning."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    adapter_name = config.get("adapter_name", "adapter")
    base_model_id = config.get("base_model_id", "Qwen/Qwen2.5-7B-Instruct")
    lora_cfg = config.get("lora", {})
    quant_cfg = config.get("quantization", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    sft_cfg = config.get("sft", {})

    if output_dir is None:
        output_dir = f"adapters/{adapter_name}"

    logger.info("=" * 60)
    logger.info("Training adapter: %s", adapter_name)
    logger.info("Base model: %s", base_model_id)
    logger.info("LoRA rank: %d, alpha: %d", lora_cfg.get("r", 32), lora_cfg.get("lora_alpha", 64))
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    # --- Quantization ---
    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    # --- Load tokenizer & model ---
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model with 4-bit quantization …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config ---
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # --- Load data ---
    train_file = data_cfg.get("train_file", "")
    eval_file = data_cfg.get("eval_file", "")
    max_samples = data_cfg.get("max_samples")

    if not train_file or not Path(train_file).exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            f"Run the data pipeline first:\n"
            f"  1. python training/data_prep/kanoon_scraper.py --domain {adapter_name}\n"
            f"  2. python training/data_prep/format_qa_pairs.py --input data/raw/{adapter_name}_judgments.jsonl --domain {adapter_name} --output {train_file}"
        )

    train_ds, eval_ds = load_dataset(train_file, eval_file, max_samples)
    logger.info("Train samples: %d, Eval samples: %d", len(train_ds), len(eval_ds) if eval_ds else 0)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=train_cfg.get("eval_steps", 100) if eval_ds else None,
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        report_to=train_cfg.get("report_to", "none"),
        seed=data_cfg.get("seed", 42),
    )

    # --- SFT Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        packing=sft_cfg.get("packing", False),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
    )

    # --- Train ---
    logger.info("Starting training …")
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # --- Save ---
    logger.info("Saving adapter to %s …", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save adapter metadata
    metadata = {
        "adapter_name": adapter_name,
        "domain": config.get("domain", ""),
        "base_model": base_model_id,
        "lora_r": lora_cfg.get("r", 32),
        "lora_alpha": lora_cfg.get("lora_alpha", 64),
        "train_samples": len(train_ds),
        "epochs": train_cfg.get("num_train_epochs", 3),
    }
    with open(Path(output_dir) / "adapter_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete! Adapter saved to %s", output_dir)
    return output_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train a QLoRA adapter for Indian legal domain")
    parser.add_argument("--config", required=True, help="Path to domain YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.epochs:
        overrides.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.learning_rate:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.batch_size:
        overrides.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size

    config = load_config(args.config, overrides if overrides else None)
    train(config, output_dir=args.output_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
