"""
Colab-optimised training wrapper.

INSTRUCTIONS:
=============
1. Open Google Colab with a T4 or A100 GPU runtime
2. Upload your training data (JSONL files) to Google Drive
3. Run this script cell by cell

This file is designed to be pasted into a Colab notebook cell by cell,
or uploaded and run as:
    !python colab_train.py --domain criminal_violent
"""

# ============================================================
# CELL 1: Install dependencies (run once)
# ============================================================
INSTALL_DEPS = """
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers peft trl accelerate
!pip install -q bitsandbytes datasets sentencepiece pyyaml
"""

# ============================================================
# CELL 2: Mount Drive and set paths
# ============================================================
MOUNT_DRIVE = """
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/legal-ai-in-main'
os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")
"""

# ============================================================
# CELL 3: Training function
# ============================================================

import time as _time


def _log(msg: str):
    ts = _time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def train_on_colab(
    domain: str = "criminal_violent",
    train_file: str | None = None,
    eval_file: str | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_seq_length: int = 2048,
    output_dir: str | None = None,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """
    Train a QLoRA adapter on Colab.

    Args:
        domain: One of the 12 legal domains (determines system prompt)
        train_file: Path to training JSONL
        eval_file: Path to eval JSONL (optional)
        epochs: Number of training epochs
        batch_size: Per-device batch size (4 for T4, 8 for A100)
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        max_seq_length: Max sequence length
        output_dir: Where to save the adapter
        base_model: HuggingFace model ID
    """
    import torch
    import trl
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    import json
    from pathlib import Path

    start_time = _time.time()

    if train_file is None:
        train_file = f"data/training/{domain}_train.jsonl"
    if output_dir is None:
        output_dir = f"adapters/{domain}"

    _log(f"{'='*60}")
    _log(f"Training adapter: {domain}")
    _log(f"Base model: {base_model}")
    _log(f"LoRA r={lora_r}, alpha={lora_alpha}")
    _log(f"Train file: {train_file}")
    _log(f"Output: {output_dir}")
    _log(f"trl version: {trl.__version__}")
    _log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        _log(f"GPU memory: {mem:.1f} GB")
    _log(f"{'='*60}")

    # ── Step 1: Quantization config ──
    _log("Step 1/7: Configuring 4-bit quantization …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Step 2: Tokenizer ──
    _log("Step 2/7: Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _log(f"  Vocab size: {len(tokenizer):,}")

    # ── Step 3: Model ──
    _log("Step 3/7: Loading base model (this takes a few minutes) …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        _log(f"  GPU memory used after loading: {alloc:.1f} GB")

    # ── Step 4: LoRA ──
    _log("Step 4/7: Applying LoRA adapters …")
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    _log(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Step 5: Data ──
    _log("Step 5/7: Loading training data …")
    train_items = []
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_items.append(json.loads(line))
    train_ds = Dataset.from_list(train_items)
    _log(f"  Train samples: {len(train_ds)}")

    eval_ds = None
    if eval_file and Path(eval_file).exists():
        eval_items = []
        with open(eval_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eval_items.append(json.loads(line))
        eval_ds = Dataset.from_list(eval_items)
        _log(f"  Eval samples: {len(eval_ds)}")

    total_steps = (len(train_ds) // (batch_size * grad_accum)) * epochs
    _log(f"  Total training steps: ~{total_steps}")

    # ── Step 6: Build trainer (version-agnostic) ──
    _log("Step 6/7: Configuring trainer …")

    common_args = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=max(1, int(total_steps * 0.03)),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=100 if eval_ds else None,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    try:
        from trl import SFTConfig
        training_args = SFTConfig(**common_args, packing=False, max_seq_length=max_seq_length)
        trainer = SFTTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
        _log("  Using SFTConfig (trl >= 0.12)")
    except (ImportError, TypeError):
        training_args = TrainingArguments(**common_args)
        trainer = SFTTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer,
            packing=False,
            max_seq_length=max_seq_length,
        )
        _log("  Using TrainingArguments + SFTTrainer kwargs (trl < 0.12)")

    # ── Step 7: Train ──
    _log("Step 7/7: Starting training …")
    _log(f"  Epochs: {epochs} | Batch: {batch_size} | Grad accum: {grad_accum} | LR: {learning_rate}")
    trainer.train()

    elapsed = _time.time() - start_time
    _log(f"Training finished in {elapsed/60:.1f} minutes")

    # ── Save ──
    _log(f"Saving adapter to {output_dir} …")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "adapter_name": domain, "base_model": base_model,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "train_samples": len(train_ds), "epochs": epochs,
        "training_time_minutes": round(elapsed / 60, 1),
    }
    with open(f"{output_dir}/adapter_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _log(f"Done! Adapter saved to {output_dir}")
    _log(f"To use: copy {output_dir}/ to your server's adapters/ directory.")

    return output_dir


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Legal domain adapter to train")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    train_on_colab(
        domain=args.domain,
        train_file=args.train_file,
        eval_file=args.eval_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        output_dir=args.output_dir,
    )
