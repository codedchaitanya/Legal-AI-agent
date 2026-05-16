"""
ILLA E2E smoke test — exercises the core research pipeline without
requiring MongoDB / Redis / Postgres / MinIO.

Tests:
  1. Base model loads (Qwen2.5-7B-Instruct in 4-bit NF4)
  2. Domain adapter activates via PEFT set_adapter
  3. QLoRA forward pass generates legal analysis
  4. Rule-based citation validator tags unverified citations
  5. (optional) Claude Sonnet synthesises adversarial arguments

Run:
    python scripts/e2e_smoke.py
    python scripts/e2e_smoke.py --domain criminal_property
    python scripts/e2e_smoke.py --no-args   # skip Claude call

Requires:
    ANTHROPIC_API_KEY   (only if --no-args is not passed)
    HF_TOKEN            (only if the base model needs gated access)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Make project importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Default test corpus
DEFAULT_CONTEXT = """
Section 103 of the Bharatiya Nyaya Sanhita 2023 (BNS) prescribes punishment
for the offence of murder. To establish a charge under Section 103, the
prosecution must prove that the accused caused the death of the victim
with one of the following mental states: (a) the intention of causing
death, (b) the intention of causing such bodily injury as the offender
knows is likely to cause the death of the person to whom the harm is
caused, (c) the intention of causing bodily injury sufficient in the
ordinary course of nature to cause death, or (d) the knowledge that the
act is so imminently dangerous that it must in all probability cause
death.

Section 35 of the BNS provides for the right of private defence of the
body, subject to the restrictions contained in Section 37. A person
exercising private defence is not guilty of an offence if the harm
caused does not exceed what is necessary for the purpose of defence
against the apprehended danger.
"""

DEFAULT_QUERY = (
    "Under BNS Section 103, what mental-state elements must the prosecution "
    "establish to secure a conviction for murder, and how does the right of "
    "private defence under BNS Section 35 affect culpability?"
)


def parse_args():
    p = argparse.ArgumentParser(description="ILLA E2E smoke test")
    p.add_argument(
        "--domain",
        default="criminal_violent",
        help="Adapter domain to activate (default: criminal_violent)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=384,
        help="Max new tokens for QLoRA generation (lower for tight VRAM)",
    )
    p.add_argument(
        "--no-args",
        action="store_true",
        help="Skip the Claude argument-generation step",
    )
    p.add_argument(
        "--context-file",
        type=Path,
        help="Path to a custom .txt file for the legal context",
    )
    p.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Override the default research query",
    )
    return p.parse_args()


def main():
    args = parse_args()

    context = DEFAULT_CONTEXT
    if args.context_file:
        context = args.context_file.read_text(encoding="utf-8")

    print(f"\n{'='*72}\nILLA E2E smoke test\n{'='*72}")
    print(f"  Domain:       {args.domain}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Arg gen:      {'OFF' if args.no_args else 'ON (Claude Sonnet)'}")
    print(f"  Query:        {args.query}")
    print()

    # ── Step 1: Load engine and activate adapter ─────────────────────────
    print(f"[1/4] Loading base model + activating adapter [{args.domain}]…")
    t0 = time.time()
    from core.reasoning.lora_engine import lora_engine

    try:
        lora_engine.activate([args.domain])
    except FileNotFoundError as e:
        print(f"  ERROR: adapter '{args.domain}' not found on disk.")
        print(f"  Available adapters: {sorted(p.name for p in Path('adapters').iterdir() if p.is_dir())}")
        print(f"  Try one of those with --domain <name>.")
        sys.exit(1)
    print(f"  ✓ loaded in {time.time() - t0:.1f}s")

    # ── Step 2: QLoRA generation ─────────────────────────────────────────
    print("\n[2/4] Running QLoRA forward pass…")
    prompt = f"""You are an expert Indian legal research assistant.
Answer using ONLY the provided legal context. Cite section numbers exactly
as they appear in the context. Do not invent citations.

=== CONTEXT ===
{context}

=== QUERY ===
{args.query}

A:"""

    t1 = time.time()
    raw = lora_engine.generate(
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=0.3,
    )
    print(f"  ✓ generated in {time.time() - t1:.1f}s")
    print("\n--- raw QLoRA answer ---")
    print(raw.strip())

    # ── Step 3: Citation validation ──────────────────────────────────────
    print("\n[3/4] Validating citations against context…")
    from core.validation.citation_validator import validate_citations

    t2 = time.time()
    validated = validate_citations(raw, context)
    print(f"  ✓ validated in {time.time() - t2:.2f}s")
    print("\n--- validated answer (unverified citations tagged) ---")
    print(validated.strip() if isinstance(validated, str) else validated)

    # ── Step 4: Argument generation (optional) ───────────────────────────
    if args.no_args:
        print("\n[4/4] Skipped (--no-args)")
    elif not os.getenv("ANTHROPIC_API_KEY"):
        print("\n[4/4] Skipped — ANTHROPIC_API_KEY not set in environment.")
        print("       export ANTHROPIC_API_KEY='sk-ant-...' to enable.")
    else:
        print("\n[4/4] Synthesising adversarial arguments via Claude Sonnet…")
        from core.reasoning.argument_generator import generate_arguments

        t3 = time.time()
        result = asyncio.run(
            generate_arguments(
                case_summary=context,
                research_context=validated if isinstance(validated, str) else raw,
                domain=args.domain,
                image_paths=[],
                side="both",
            )
        )
        print(f"  ✓ generated in {time.time() - t3:.1f}s")
        print("\n--- structured arguments ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'='*72}\n✓ E2E smoke test passed\n{'='*72}\n")


if __name__ == "__main__":
    main()
