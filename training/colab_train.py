"""
Colab-optimised training wrapper.

INSTRUCTIONS:
=============
1. Open Google Colab with a T4 or A100 GPU runtime
   (Runtime → Change runtime type → Hardware accelerator → T4 GPU)
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
# CELL 3: Sanity check — make sure GPU is attached
# ============================================================
GPU_CHECK = """
import torch
assert torch.cuda.is_available(), (
    "No GPU detected! Go to Runtime → Change runtime type → "
    "Hardware accelerator → T4 GPU, then restart."
)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
!nvidia-smi
"""

# ============================================================
# CELL 4: Training function
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
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, TrainerState, TrainerControl, PrinterCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import json
    from pathlib import Path

    class ProgressCallback(TrainerCallback):
        def __init__(self, total_steps: int):
            self.total_steps = total_steps
            self.start_time = None
            self.epoch_start = None
            self.current_epoch = 0

        def _gpu_mem(self) -> str:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f"{alloc:.1f}/{reserved:.1f} GB"

        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
            self.start_time = time.time()
            print(f"\n{'='*60}")
            print(f"  Training started — {self.total_steps} total steps")
            print(f"  GPU memory at start: {self._gpu_mem()} (alloc/reserved)")
            print(f"{'='*60}\n")

        def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
            self.current_epoch += 1
            self.epoch_start = time.time()
            print(f"\n--- Epoch {self.current_epoch}/{args.num_train_epochs} started ---")

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kw):
            elapsed = time.time() - self.epoch_start
            print(f"--- Epoch {self.current_epoch}/{args.num_train_epochs} done in {elapsed/60:.1f} min  |  GPU mem: {self._gpu_mem()} ---\n")

        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
            if not logs or state.global_step == 0:
                return
            elapsed = time.time() - self.start_time
            pct = state.global_step / self.total_steps
            eta_s = (elapsed / pct - elapsed) if pct > 0 else 0
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)

            loss = logs.get("loss", logs.get("train_loss", "—"))
            lr = logs.get("learning_rate", "—")
            grad = logs.get("grad_norm", "—")

            loss_str = f"{loss:.4f}" if isinstance(loss, float) else loss
            lr_str = f"{lr:.2e}" if isinstance(lr, float) else lr
            grad_str = f"{grad:.3f}" if isinstance(grad, float) else grad

            print(
                f"[{bar}] {state.global_step}/{self.total_steps} ({100*pct:.1f}%)  "
                f"loss={loss_str}  lr={lr_str}  grad={grad_str}  "
                f"elapsed={elapsed/60:.1f}m  ETA={eta_s/60:.1f}m  "
                f"GPU={self._gpu_mem()}"
            )

    if train_file is None:
        train_file = f"data/training/{domain}_train.jsonl"
    if output_dir is None:
        output_dir = f"adapters/{domain}"

    # Hard-fail early if no GPU — QLoRA on CPU will not work.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. QLoRA 4-bit training requires a GPU. "
            "In Colab: Runtime → Change runtime type → T4 GPU, then restart."
        )

    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    # Pick precision based on hardware (T4 = fp16, A100 = bf16)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{'='*60}")
    print(f"Training adapter: {domain}")
    print(f"Base model: {base_model}")
    print(f"LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"Train file: {train_file}")
    print(f"Output: {output_dir}")
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")
    print(f"Precision: {'bf16' if use_bf16 else 'fp16'}")
    print(f"{'='*60}")

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # In TRL 0.24, max sequence length is driven by the tokenizer, not SFTConfig.
    tokenizer.model_max_length = max_seq_length

    # Model
    print("Loading model with 4-bit quantization …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
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
    # NOTE: max_seq_length was removed from SFTConfig in newer TRL versions.
    # It is now controlled via tokenizer.model_max_length (set above).
    training_args = SFTConfig(
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
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        packing=False,
    )

    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
        callbacks=[ProgressCallback(total_steps)],
    )

    print(f"Steps per epoch: {steps_per_epoch}  |  Total steps: {total_steps}")
    print("Starting training …")
    train_result = trainer.train()

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Runtime: {train_result.metrics.get('train_runtime', 0)/60:.1f} min")
    print(f"  Samples/sec: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
    print(f"  Final loss: {train_result.metrics.get('train_loss', '—'):.4f}" if isinstance(train_result.metrics.get('train_loss'), float) else "  Final loss: —")
    print(f"{'='*60}\n")

    # Save
    print(f"Saving adapter to {output_dir} …")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "adapter_name": domain,
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_samples": len(train_ds),
        "epochs": epochs,
        "max_seq_length": max_seq_length,
        "precision": "bf16" if use_bf16 else "fp16",
    }
    with open(f"{output_dir}/adapter_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Adapter saved to {output_dir}")
    print(f"To use: copy {output_dir}/ to your server's adapters/ directory.")

    return output_dir


# ============================================================
# Fast multi-domain trainer (loads model once)
# ============================================================

ALL_DOMAINS = [
    "criminal_violent", "criminal_property", "kidnapping_trafficking",
    "sexual_offences", "land_property", "family_matrimonial",
    "constitutional", "corporate_commercial", "labour_employment",
    "cyber_digital", "tax_fiscal", "civil_general",
]

def train_all_adapters(
    domains: list = None,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_seq_length: int = 1024,
    early_stopping_patience: int = 3,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """Train all domain adapters with a single model load — much faster than calling
    train_on_colab() per domain, which reloads the 7B model each time."""
    import torch, time, json, os
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, TrainerState, TrainerControl, PrinterCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from pathlib import Path

    if domains is None:
        domains = ALL_DOMAINS

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"{'='*60}")
    print(f"Training {len(domains)} adapters on {gpu_name} ({gpu_total_gb:.1f} GB)")
    print(f"Domains: {', '.join(domains)}")
    print(f"{'='*60}\n")

    # ── Load tokenizer + model ONCE ──────────────────────────
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    print("Loading model with 4-bit quantization …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    print("Model loaded. Starting domain loop …\n")

    total_start = time.time()
    results = {}

    for idx, domain in enumerate(domains):
        domain_start = time.time()
        train_file = f"data/training/{domain}_train.jsonl"
        output_dir = f"adapters/{domain}"

        if not Path(train_file).exists():
            print(f"[{idx+1}/{len(domains)}] SKIP {domain} — {train_file} not found")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(domains)}] Training: {domain}")
        print(f"{'='*60}")

        # Load data
        items = [json.loads(l) for l in open(train_file, encoding="utf-8") if l.strip()]
        train_ds = Dataset.from_list(items)
        print(f"Samples: {len(train_ds)}")

        # Fresh LoRA adapter
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        if idx == 0:
            model = get_peft_model(base, lora_config, adapter_name=domain)
        else:
            model.add_adapter(domain, lora_config)
            model.set_adapter(domain)
            model.enable_adapter_layers()

        trainable, total = model.get_nb_trainable_parameters()
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
        total_steps = steps_per_epoch * epochs

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=1,          # log every step so early stopping is per-step
            save_steps=500,
            save_total_limit=1,
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to="none",
            packing=False,
        )

        class _Cb(TrainerCallback):
            def __init__(self, total, patience, print_every=5):
                self.total = total; self.t0 = None; self.ep = 0; self.ep_t = None
                self.patience = patience
                self.print_every = print_every
                self.best_loss = float("inf")
                self.bad_steps = 0

            def on_train_begin(self, args, state, control, **kw):
                self.t0 = time.time()

            def on_epoch_begin(self, args, state, control, **kw):
                self.ep += 1; self.ep_t = time.time()
                print(f"  Epoch {self.ep}/{args.num_train_epochs} started")

            def on_epoch_end(self, args, state, control, **kw):
                print(f"  Epoch {self.ep}/{args.num_train_epochs} done in {(time.time()-self.ep_t)/60:.1f} min")

            def on_log(self, args, state, control, logs=None, **kw):
                if not logs or state.global_step == 0: return
                loss = logs.get("loss", None)

                # Early stopping checked every step (logging_steps=1)
                if isinstance(loss, float):
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.bad_steps = 0
                    else:
                        self.bad_steps += 1
                        if self.bad_steps >= self.patience:
                            print(f"  ⏹ Early stop step {state.global_step}: loss={loss:.4f} > best={self.best_loss:.4f}")
                            control.should_training_stop = True
                            return

                # Print progress every N steps to avoid spam
                if state.global_step % self.print_every != 0:
                    return
                elapsed = time.time() - self.t0
                pct = state.global_step / self.total
                eta = (elapsed / pct - elapsed) if pct > 0 else 0
                loss_s = f"{loss:.4f}" if isinstance(loss, float) else "—"
                bar = "█" * int(20*pct) + "░" * (20 - int(20*pct))
                alloc = torch.cuda.memory_allocated()/1e9
                print(f"  [{bar}] {state.global_step}/{self.total} loss={loss_s} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m GPU={alloc:.1f}GB")

        trainer = SFTTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, processing_class=tokenizer,
            max_seq_length=max_seq_length, callbacks=[_Cb(total_steps, early_stopping_patience)],
        )
        print(f"Steps/epoch: {steps_per_epoch}  |  Total: {total_steps}")
        result = trainer.train()

        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        json.dump({
            "adapter_name": domain, "base_model": base_model,
            "lora_r": lora_r, "lora_alpha": lora_alpha,
            "train_samples": len(train_ds), "epochs": epochs,
        }, open(f"{output_dir}/adapter_metadata.json", "w"), indent=2)

        elapsed_domain = (time.time() - domain_start) / 60
        results[domain] = {"runtime_min": round(elapsed_domain, 1), "loss": result.metrics.get("train_loss")}
        print(f"  Done in {elapsed_domain:.1f} min → saved to {output_dir}")

        # Free LoRA weights before next domain
        model.delete_adapter(domain)
        torch.cuda.empty_cache()

    total_min = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"All {len(results)}/{len(domains)} adapters trained in {total_min:.1f} min")
    for d, r in results.items():
        print(f"  {d}: {r['runtime_min']} min  loss={r['loss']:.4f}" if isinstance(r['loss'], float) else f"  {d}: {r['runtime_min']} min")
    print(f"{'='*60}")
    return results


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
    parser.add_argument("--max-seq-length", type=int, default=2048)
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
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )