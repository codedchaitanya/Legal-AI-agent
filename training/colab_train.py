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
!pip install -q transformers==4.57.6 peft==0.17.1 trl==0.24.0 accelerate==1.10.1
!pip install -q bitsandbytes==0.42.0 datasets sentencepiece pyyaml
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    import json
    from pathlib import Path

    if train_file is None:
        train_file = f"data/training/{domain}_train.jsonl"
    if output_dir is None:
        output_dir = f"adapters/{domain}"

    print(f"{'='*60}")
    print(f"Training adapter: {domain}")
    print(f"Base model: {base_model}")
    print(f"LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"Train file: {train_file}")
    print(f"Output: {output_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    print("Loading model with 4-bit quantization …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Data
    print("Loading data …")
    train_items = []
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_items.append(json.loads(line))
    train_ds = Dataset.from_list(train_items)
    print(f"Train samples: {len(train_ds)}")

    eval_ds = None
    if eval_file and Path(eval_file).exists():
        eval_items = []
        with open(eval_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eval_items.append(json.loads(line))
        eval_ds = Dataset.from_list(eval_items)
        print(f"Eval samples: {len(eval_ds)}")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
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

    # Train
    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, packing=False,
        max_seq_length=max_seq_length,
    )

    print("Starting training …")
    trainer.train()

    # Save
    print(f"Saving adapter to {output_dir} …")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "adapter_name": domain, "base_model": base_model,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "train_samples": len(train_ds), "epochs": epochs,
    }
    with open(f"{output_dir}/adapter_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Adapter saved to {output_dir}")
    print(f"To use: copy {output_dir}/ to your server's adapters/ directory.")

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
