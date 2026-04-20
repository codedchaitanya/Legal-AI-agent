"""
Retrain an existing LoRA adapter on CLAT MCQ data to improve MCQ benchmark scores.
Loads the saved adapter, mixes CLAT MCQ samples with original Q&A training data,
and continues training with a lower LR for 2 epochs.

Usage (from repo root):
    python training/clat-data/retrain_clat.py --domain cyber_digital
    python training/clat-data/retrain_clat.py --domain civil_general --epochs 3 --mix_ratio 0.4
"""

import argparse, json, os, time
from pathlib import Path

def retrain_adapter_on_clat(
    domain: str,
    epochs: int = 2,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 1e-4,          # lower than original 2e-4
    max_seq_length: int = 512,             # MCQ prompts are short — 512 is enough
    mix_ratio: float = 0.3,               # fraction of original Q&A data to mix in
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    clat_dir: str = "training/clat-data",
    adapter_base_dir: str = "adapters",
    use_flash_attention: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from peft import PeftModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import random, math

    clat_file    = Path(clat_dir)  / f"{domain}_clat.jsonl"
    orig_file    = Path("data/training") / f"{domain}_train.jsonl"
    adapter_dir  = Path(adapter_base_dir) / domain
    adapter_path = adapter_dir / domain    # PEFT saves adapter in a subdirectory named after adapter

    if not clat_file.exists():
        raise FileNotFoundError(f"CLAT data not found: {clat_file}\nRun download_and_split.py first.")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_dir}\nTrain the base adapter first.")

    print(f"\n{'='*60}")
    print(f"  Retraining: {domain}")
    print(f"  CLAT file : {clat_file}")
    print(f"  Adapter   : {adapter_dir}")
    print(f"{'='*60}\n")

    use_bf16       = torch.cuda.is_bf16_supported()
    compute_dtype  = torch.bfloat16 if use_bf16 else torch.float16

    # ── Load data ─────────────────────────────────────────────────────────────
    clat_samples = [json.loads(l) for l in open(clat_file, encoding="utf-8") if l.strip()]
    orig_samples = []
    if orig_file.exists() and mix_ratio > 0:
        orig_all = [json.loads(l) for l in open(orig_file, encoding="utf-8") if l.strip()]
        n_orig   = min(len(orig_all), math.ceil(len(clat_samples) * mix_ratio / (1 - mix_ratio)))
        orig_samples = random.sample(orig_all, n_orig)

    combined = clat_samples + orig_samples
    random.shuffle(combined)
    train_ds = Dataset.from_list(combined)
    print(f"  CLAT={len(clat_samples)}  Q&A-mix={len(orig_samples)}  total={len(train_ds)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length
    tokenizer.padding_side = "right"

    # ── Base model ────────────────────────────────────────────────────────────
    print("  Loading base model (4-bit) …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    fa2_kwargs = {"attn_implementation": "flash_attention_2"} if use_flash_attention else {}
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, **fa2_kwargs,
    )
    base.config.use_cache = False

    # ── Load existing adapter ─────────────────────────────────────────────────
    peft_path = adapter_path if adapter_path.exists() else adapter_dir
    print(f"  Adapter: {peft_path}")
    model = PeftModel.from_pretrained(base, str(peft_path), adapter_name=domain, is_trainable=True)
    model.train()
    model.enable_adapter_layers()
    model.enable_input_require_grads()   # required: gradient_checkpointing + PEFT needs this

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Training args ─────────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
    total_steps     = steps_per_epoch * epochs
    log_every       = max(1, total_steps // 20)
    print(f"  Steps: {steps_per_epoch}/epoch × {epochs} = {total_steps}  (log every {log_every})\n")

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_seq_length=max_seq_length,             # set here, not in SFTTrainer — avoids deprecation warn
        logging_steps=log_every,
        save_steps=total_steps + 1,                # only save at end
        save_total_limit=1,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        packing=False,
    )

    # ── Progress callback ─────────────────────────────────────────────────────
    class _Cb(TrainerCallback):
        def __init__(self, total):
            self.total = total; self.t0 = None
        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs: return
            loss = logs.get("loss")
            if isinstance(loss, float):
                elapsed = time.time() - self.t0
                pct     = state.global_step / self.total
                eta     = (elapsed / pct - elapsed) if pct > 0 else 0
                print(f"    [{state.global_step:>4}/{self.total}] loss={loss:.4f}  "
                      f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, processing_class=tokenizer,
        callbacks=[_Cb(total_steps)],
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = (time.time() - t0) / 60

    # ── Save updated adapter ───────────────────────────────────────────────────
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta_path = adapter_dir / "adapter_metadata.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    loss_val = result.metrics.get("train_loss")
    meta.update({
        "clat_retrain_samples": len(clat_samples),
        "clat_mix_orig_samples": len(orig_samples),
        "clat_retrain_epochs": epochs,
        "clat_retrain_lr": learning_rate,
        "clat_train_loss": loss_val,
    })
    json.dump(meta, open(meta_path, "w"), indent=2)

    print(f"\n  Done: {elapsed:.1f} min  |  loss={loss_val:.4f}  →  {adapter_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",    required=True)
    parser.add_argument("--epochs",    type=int,   default=2)
    parser.add_argument("--batch_size",type=int,   default=4)
    parser.add_argument("--grad_accum",type=int,   default=4)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--mix_ratio", type=float, default=0.3)
    parser.add_argument("--flash",     action="store_true")
    args = parser.parse_args()

    retrain_adapter_on_clat(
        domain=args.domain,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        mix_ratio=args.mix_ratio,
        use_flash_attention=args.flash,
    )
