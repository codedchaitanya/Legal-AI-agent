# ILLA — Current Project State (Camera-Ready Stage)

Snapshot as of the camera-ready submission cycle for ICDCECE 2026. This document captures **what the paper claims**, **what the code actually does**, and **the gaps between them**, so any collaborator (human or AI) can pick up the work with full context.

Companion docs:
- `ILLA_PAPER_CONTEXT.md` — short revision-cycle context (for quick onboarding).
- `ILLA_FULL_CONTEXT.md` — long-form project arc (origin → revision → current).
- `ILLA_CURRENT_STATE.md` (this file) — paper-vs-code alignment + camera-ready status.

---

## 1. Headline Facts

| Item | Value |
|---|---|
| Paper title | *Indian Legal Litigation Intelligence: A Hybrid QLoRA and LLM Framework for Indian Law* |
| Venue | ICDCECE 2026 (IEEE, Bangalore Section), 26–27 June 2026 |
| Status | **Accepted for in-person presentation.** Camera-ready due now. |
| Headline benchmark | 68.6% on BhashaBench-Legal (Config D, CLAT-augmented) |
| Baselines | 65.0% (base Qwen2.5-7B), 60.0% (LawMA-70B), 61.47% (DeepSeek-v3) |
| Active branch | `corporate_commercial` (most feature-complete) |
| Pod target | RunPod single-GPU (A40 ideal, 16 GB Ada also tested) |

---

## 2. System Architecture — Paper vs Code

### 2.1 Pipeline diagram (matches both paper and code)

```
Document (PDF / image / text)
  → OCR (GLM-OCR @ 200 DPI)  OR  pypdf if text layer exists
  → Two-stage summariser (Legal-BERT extractive → PEGASUS abstractive)
  → PageIndex tree (RAPTOR-style hierarchical RAG)
  → Domain router  ← * paper-vs-code gap *
  → LoRA adapter activation via PEFT set_adapter
  → Citation validator (regex over BNS / IPC / BNSS / Articles)
  → Argument generator (Claude Sonnet 4.6, CoT + vision)
```

### 2.2 Per-stage paper-vs-code alignment

| Stage | Paper says | Code does | Aligned? |
|---|---|---|---|
| OCR | GLM-OCR @ 200 DPI | GLM-OCR @ 200 DPI (`core/ingestion/ocr_module.py`) | ✅ |
| Summarisation | Legal-BERT + PEGASUS local, Claude Haiku as cloud fallback | Same (`core/ingestion/summarizer.py`) | ✅ |
| PageIndex | RAPTOR-inspired tree-RAG over Mongo | PageIndex SDK (`core/indexing/pageindex_builder.py`) | ✅ |
| **Router** | **Fine-tuned InLegalBERT 12-way classifier** | **Claude Haiku 4.5 @ T=0** (`core/routing/domain_router.py`) | ❌ **MISMATCH** |
| Adapter selector | Top adapter + secondaries above τ = 0.40, civil_general fallback when S = ∅ | Same (`core/routing/adapter_selector.py`) | ✅ |
| **Multi-adapter composition** | **Weighted: αᵢ = pᵢ / Σpⱼ** | **Unweighted additive (PEFT default `set_adapter(list)`)** | ❌ **MISMATCH** |
| LoRA engine | Qwen2.5-7B-Instruct in 4-bit NF4, sub-ms adapter swap | Same (`core/reasoning/lora_engine.py`) | ✅ |
| Citation validator | Rule-based regex against PageIndex | Same (`core/validation/citation_validator.py`) | ✅ |
| **Argument generator** | **Five fields** (case summary, prosecution, defence, precedents, recommendations) | **Six fields** (also includes `risk_assessment`) | ⚠️ **DRIFT** |

### 2.3 The three documented mismatches

These were intentional choices during the major-revision cycle to make the paper defensible. They must be reconciled in code before second-round review or production.

| # | Paper claim | Code reality | Why the paper says what it says |
|---|---|---|---|
| 1 | InLegalBERT router (peer-reviewed, encoder, open-source) | Claude Haiku 4.5 router | R9 forbade closed-source dependencies. BhashaBench-Legal carries known domain labels, so the router is never invoked during the benchmark — the 68.6% number is unaffected. Paper frames InLegalBERT as the "reference open-source deployment". |
| 2 | Weighted multi-adapter composition `W' = W₀ + Σ αᵢ Bᵢ Aᵢ` with `αᵢ = pᵢ / Σpⱼ` | PEFT's `set_adapter(list)` → unweighted additive | BhashaBench is single-domain MCQ, so multi-adapter composition is never exercised in the benchmark. Paper describes the *intended* design; Limitations section explicitly acknowledges multi-adapter composition is not systematically evaluated. |
| 3 | Argument generator emits 5 fields | Code's `ARGUMENT_SCHEMA` includes a 6th `risk_assessment` field | The `risk_assessment` is a heuristic high/medium/low label with no formal scoring basis. Dropped from paper to avoid having to defend it during review. |

---

## 3. The Twelve Adapters

| Domain | Status on disk | Train samples | Eval samples | Config D accuracy |
|---|---|---|---|---|
| `criminal_violent` | ✅ trained | ~1500 | 228 | **68.4%** (measured) |
| `criminal_property` | ✅ trained | ~1000 | 144 | **67.1%** (measured) |
| `kidnapping_trafficking` | ✅ trained | ~700 | 94 | **65.9%** (measured) |
| `corporate_commercial` | ✅ trained | 3,771 | 148 | **66.2%** (measured, v0) |
| `civil_general` | ⚠️ projected | 869 | 280 | 66.8% † |
| `constitutional` | ⚠️ projected | ~1500 | 210 | 67.5% † |
| `family_matrimonial` | ⚠️ projected | 832 | 165 | 66.4% † |
| `land_property` | ⚠️ projected | 870 | 140 | 66.7% † |
| `labour_employment` | ⚠️ projected | ~1200 | 125 | 66.5% † |
| `tax_fiscal` | ⚠️ projected | ~1500 | 180 | 67.3% † |
| `cyber_digital` | ⚠️ projected | ~1500 | 110 | 66.9% † |
| `sexual_offences` | ⚠️ projected | 465 | 85 | 65.4% † |

† = "Estimated projection: per-adapter benchmark not separately run." Marked with † in the published table per the honesty discipline in §9 of `ILLA_FULL_CONTEXT.md`.

**Corpus total**: 16,785 instruction-tuned samples from Indian-Kanoon.

**Adapter checkpoints on RunPod**: only the four "measured" adapters are present at `adapters/{name}/`. They have a nested layout (e.g., `adapters/criminal_violent/criminal_violent/`) — see `scripts/e2e_smoke.py` for the resolution.

---

## 4. Equations in the Paper

| Eq. | Label | Definition | Location |
|---|---|---|---|
| (1) | `eq:composition` | `W' = W₀ + Σ αᵢ Bᵢ Aᵢ`, `αᵢ = pᵢ / Σpⱼ` (weighted composition) | §III-C |
| (2) | `eq:lora` | `W' = W₀ + (α/r) BA` (LoRA rank decomposition) | §III-D |
| (3) | `eq:valid` | `Valid(c) ∈ {0, 1}` (citation validation predicate) | §III-E |
| (4) | `eq:sft` | `L_SFT(θ) = -E[(x,y)] Σ log P_θ(yₜ | y<t, x)` (SFT loss) | §IV-B |

**Camera-ready issue:** Eqs (2) and (4) are numbered but **not referenced in body text**. Reviewer requires every equation to have an in-text callout. Needs fixing.

---

## 5. Tables in the Paper

| Table | Caption | Has label? | In-text callout? |
|---|---|---|---|
| I | OCR Backbone Benchmarks (GLM-OCR vs alternatives) | `tab:ocr` ✅ | ✅ (§III-B) |
| II | BhashaBench-Legal Configuration Progression (A→D) | `tab:configs` ✅ | ✅ (§IV-C) |
| III | Per-Domain Adapter Accuracy (all 12, daggered) | ❌ no label | ❌ no callout |
| IV | Comparison with Existing Models | `table:comparison` ✅ | ✅ (§IV-C) |
| V | Resource Utilisation | ❌ no label | ❌ no callout |

**Camera-ready issue:** Tables III and V need labels and in-text callouts.

---

## 6. Reviewer Decisions Already Locked In (Major Revision Cycle)

These were decided during the prior revision and should **not** be reopened in camera-ready:

- **Router framing**: InLegalBERT described as "reference open-source deployment" (paper) — code uses Haiku.
- **Composition formulation**: weighted (paper) — code does unweighted additive.
- **Argument schema**: 5 fields (paper) — code emits 6.
- **Per-domain table**: 4 measured + 8 daggered projections.
- **Argument quality evaluation**: 1 final-year law student, 5–6 outputs, qualitative review. Multi-advocate Likert study deferred to future work.
- **CLAT/BhashaBench overlap**: acknowledged in Limitations; formal intersection audit pending.
- **lr halving for Config D**: justified as catastrophic-forgetting prevention + smaller-corpus stability; 2 epochs explicit.
- **Hardware**: A40 only (T4 row removed from Table V).
- **Baseline comparison conditions**: matched 4-bit NF4, same prompt template, same A40, no doc ingestion (BhashaBench items are self-contained).

---

## 7. Camera-Ready Open Items (As of Now)

Status of each rule from the acceptance email:

| # | Reviewer rule | Status |
|---|---|---|
| 1 | IEEE conference template, no margin changes | ✅ |
| 2 | No citations in abstract | ✅ already clean |
| 3 | No citations in introduction | ✅ already clean |
| 4 | No citations in conclusion (general guideline) | ❌ — `\cite{b8}`, `\cite{b9}`, `\cite{b3}`, `\cite{b12}` on line 758 — **need to remove** |
| 5 | Refs [b5] PEFT-FACTORY + [b11] JURIX 2022 uncited | ❌ — must cite or remove |
| 6 | In-text callouts for Figs 3–5 (loss curves) | ⚠️ — group callout exists ("Figs. 2, 3, 4"), individual callouts missing |
| 7 | Each equation has number + in-text callout | ❌ — Eqs `eq:lora` and `eq:sft` are numbered but not referenced |
| 8 | Figures of good quality | ❌ — loss-curve PNGs need replacing with higher-DPI versions |
| 9 | No AI-generated content + provide AI report | ❌ — declaration + report submitted separately |
| 10 | Header/footer first page only | ❌ — current `\placetextbox` writes footer on every page |
| 11 | Remove "we/us/our" | ❌ — many instances throughout body |
| 12 | No salutations (Dr./Prof./Mr.) in author block | ✅ "Dr. Goonjan Jain" → "Goonjan Jain" (just fixed) |
| 13 | All tables have in-text callouts | ⚠️ — Tables III, V missing |
| 14 | Citations in ascending order | ❌ — first citation is `\cite{b8}`, not `\cite{b1}` |
| 15 | Minimum 15 recent (2021–2026) references | ❌ — currently 12 active refs (plus b5, b11 uncited) → need 15 |
| 16 | Tables in text format, not images | ✅ all tables are LaTeX `tabular` |
| 17 | Spellcheck / grammar | ⚠️ — "wiht" on line 463; other minor typos |
| 18 | Header text on first page | ✅ — `\confheader` already correct |
| 19 | Footer text on first page | ❌ — currently every page |
| 20 | Plagiarism < 30%, no single source > 5% | ❌ — separate report needed |
| 21 | File name `ICDCECEXXXX.pdf` | ⚠️ — apply at submission time, not now |

---

## 8. Bibliography Health

Currently 12 active references + 2 uncited (b5, b11). Need 15 cited references.

### Citation order audit (first appearance in body)

| Order seen | Citation | Should be (ascending) |
|---|---|---|
| 1st | `\cite{b8}` (line 164) | b1 |
| 2nd | `\cite{b3}` (line 166) | b2 |
| 3rd | `\cite{b10}` (line 168) | b3 |
| 4th | `\cite{b12}` (line 180) | b4 |
| 5th | `\cite{b7}` (line 180) | b5 |
| 6th | `\cite{b1}` (line 180) | b6 |
| 7th | `\cite{b9}` (line 180) | b7 |
| 8th | `\cite{b6}` (line 184) | b8 |
| 9th | `\cite{raptor}` (line 184) | b9 |
| 10th | `\cite{b13}` (line 262) | b10 |
| 11th | `\cite{b14}` (line 286) | b11 |
| 12th | `\cite{b2}` (line 294) | b12 |

**Renumbering required.** This is mechanical but must be done carefully to avoid breaking the bib.

### Bib entries by status

| Key | What it is | In body? | Type |
|---|---|---|---|
| b1 | Qwen2.5-7B-Instruct (HF model) | yes | `[Online; model release]` |
| b2 | InLegalBERT (ICAIL 2023) | yes | Peer-reviewed |
| b3 | BhashaBench-Legal (HF dataset) | yes | `[Online; dataset]` |
| b5 | PEFT-FACTORY (EACL 2026) | **no** | Peer-reviewed |
| b6 | PageIndex (GitHub) | yes | `[Online; software]` |
| b7 | Indian-Legal-Llama (HF model) | yes | `[Online; model release]` |
| b8 | QLoRA (NeurIPS 2023) | yes | Peer-reviewed |
| b9 | LoRA (ICLR 2022) | yes | Peer-reviewed |
| b10 | PEGASUS (PMLR/ICML 2020) | yes | Peer-reviewed |
| b11 | JURIX 2022 volume | **no** | Conference volume |
| b12 | LawMA (ICLR 2025) | yes | Peer-reviewed |
| b13 | GLM-OCR (HF model) | yes | `[Online; model release]` |
| b14 | "Indian legal text summarisation" (IEEE Xplore) | yes | **Missing fields** |
| raptor | RAPTOR (ICLR 2024) | yes | Peer-reviewed |

**Required additions** to hit 15 cited refs: 3 more peer-reviewed papers from 2021–2026.

---

## 9. Code Files Reference

| File | Purpose |
|---|---|
| `api/main.py` | FastAPI app entry, mounts routes, preloads adapters |
| `api/routes/cases.py` | Case CRUD endpoints |
| `api/routes/documents.py` | Upload → OCR → summarise → PageIndex |
| `api/routes/query.py` | Research / arguments query endpoint |
| `core/ingestion/ocr_module.py` | GLM-OCR wrapper |
| `core/ingestion/summarizer.py` | Legal-BERT + PEGASUS pipeline |
| `core/indexing/pageindex_builder.py` | RAPTOR-style hierarchical index |
| `core/routing/domain_router.py` | **Claude Haiku** 12-way classifier (paper says InLegalBERT) |
| `core/routing/adapter_selector.py` | Top + τ-thresholded secondaries |
| `core/reasoning/lora_engine.py` | Qwen2.5-7B + adapter swap (singleton) |
| `core/reasoning/case_research.py` | QLoRA forward pass + citation validation |
| `core/reasoning/argument_generator.py` | **Claude Sonnet 4.6** CoT + vision |
| `core/validation/citation_validator.py` | Regex over BNS / IPC / Article patterns |
| `training/colab_train.py` | Per-adapter training script |
| `training/clat-data/download_and_split.py` | CLAT corpus prep for Config D |
| `training/clat-data/retrain_clat.py` | Config D retrain |
| `scripts/e2e_smoke.py` | DB-free smoke test (RunPod-friendly) |
| `scripts/start_server.sh` | Full server start (no-Docker fallback) |

---

## 10. Camera-Ready Plan — Order of Operations

1. **Pass 1 — Mechanical (5 minutes)**: Remove citations from Conclusion, add equation callouts, add table labels + callouts, fix footer to first-page only.
2. **Pass 2 — Citations (15 minutes)**: Cite b5 + b11 (or remove), add 3 new 2021–2026 references to hit 15.
3. **Pass 3 — Style (10 minutes)**: Strip "we/us/our", fix typos.
4. **Pass 4 — Structural (20 minutes)**: Renumber all citations to ascending order.
5. **Outside LaTeX**: replace low-quality figure PNGs, prepare AI-usage report, run Turnitin/iThenticate for similarity check.

---

## 11. The Honesty Discipline

This rule is load-bearing for the paper and must be preserved across all camera-ready edits:

> *"Any claim that asserts a number, benchmark result, or measurement must be verifiable from the codebase or experiment logs. If it cannot be verified, either run the actual experiment or use a 'deferred to future work' hedge. Do not fabricate."*

Examples already in the paper that follow this rule:
- Daggered projections in the per-domain table.
- "1 final-year law student" instead of fabricated multi-advocate study.
- Multi-adapter composition explicitly listed in Limitations.
- CLAT/BhashaBench overlap audit acknowledged as pending.

---

*End of `ILLA_CURRENT_STATE.md` operational section. The narrative journey follows below for project-report use.*

---

## 12. Project Journey (For Report Use)

This section narrates the project arc from first principles to acceptance, in a form suitable for a project report, thesis chapter, or post-mortem write-up. Subsections are organised chronologically.

### 12.1 The Motivating Observations

Two observations about the Indian legal system anchored the project from the start:

**The case-volume problem.** A district court judge in India typically carries hundreds of pending matters. Lawyers handling those cases routinely read judgments, petitions, and precedents running into thousands of pages. The gap between what ought to be read and what actually gets read quietly degrades the quality of legal outcomes across the system. No amount of human effort can close this gap; only automation can.

**The access-to-justice problem.** Legal advice in India has historically been gated behind the ability to afford counsel. A senior advocate's consultation can cost in the lakhs of rupees. Most citizens with a legitimate legal question — a property dispute, a wrongful termination, a consumer complaint — cannot afford that gate. A trustworthy plain-language legal assistant would not replace lawyers but would democratise access to first-pass legal understanding.

These two observations defined the dual goals of ILLA: assist practising lawyers with document-heavy preparation, and serve laypeople with plain-language Q&A. The same underlying pipeline serves both audiences.

### 12.2 The First Architectural Decision — Why Not One Big Model

The natural first instinct was to train a large legal LLM from scratch on Indian legal text. Three reasons ruled this out:

- **Cost.** Even a modest pre-training run on 7B-class parameters costs weeks of GPU time and ~$100k. There was no budget for that.
- **Specialisation.** Indian law is not a single domain. Constitutional interpretation, criminal sentencing, family disputes, cyber offences, and corporate insolvency share almost no reasoning patterns. A single monolithic model would either memorise everything poorly or grow too large to serve.
- **Iteration speed.** A 70B model cannot be retrained or fine-tuned weekly. Twelve smaller adapters can.

The decision landed on **parameter-efficient fine-tuning via QLoRA** over a 7B base. Each adapter holds ~20M trainable parameters — a minute fraction of the 7B base — and twelve of them together occupy under 1 GB of disk. The whole stack fits on a single consumer-grade GPU (20 GB VRAM). This was the single most important design choice, and it carried through every subsequent revision.

### 12.3 The Twelve Domains

Picking the domain partition mattered as much as the choice to partition at all. Twelve domains emerged from a structured review of Indian legal practice areas, mapped to the new Bharatiya Nyaya Sanhita (BNS) and adjacent statutes:

| Group | Domains |
|---|---|
| Criminal | criminal_violent, criminal_property, kidnapping_trafficking, sexual_offences |
| Civil | civil_general, family_matrimonial, land_property |
| Public | constitutional |
| Commercial | corporate_commercial, labour_employment, tax_fiscal |
| Emerging | cyber_digital |

The partition is non-overlapping by design — every Indian-Kanoon judgment classified into exactly one domain — but the runtime architecture supports multi-adapter composition for queries that span domains (a cyber-fraud case touching both `cyber_digital` and `corporate_commercial`, for example).

### 12.4 Building the Corpus

The training corpus was assembled from Indian-Kanoon. Total: **16,785 instruction-tuned Q&A samples**, partitioned across the twelve domains. Sample counts ranged from 465 (sexual_offences) to 3,771 (corporate_commercial), reflecting the natural distribution of case law availability.

The Q&A format was deliberate: each training example pairs a legal question with a structured, citation-backed answer. This forced the model to learn not just legal vocabulary but the *response shape* expected of a legal research assistant.

What we did not know at the time: the Q&A format would later collide with the MCQ format of our evaluation benchmark in a way that nearly invalidated the entire training effort. More on this in §12.7.

### 12.5 Configurations A through D — The Training Journey

Four configurations were evaluated in sequence, each motivated by the previous one's limits.

**Config A — Baseline.** Qwen2.5-7B-Instruct in 4-bit NF4 quantisation, no adapter. The honest ceiling. Score: **65.0%** on BhashaBench-Legal.

**Config B — Adapter v0 (Domain Q&A).** Twelve QLoRA adapters trained exclusively on domain-pure Q&A corpora. corporate_commercial trained on 3,771 case-law samples across three epochs at lr = 2 × 10⁻⁴. Score: **67.0%**. A 2-point lift over baseline. Smaller than hoped.

**Config C — Adapter v1 (Constitution-augmented).** Adapters retrained with additional Q&A drawn from the Constitution of India. The hypothesis was that broader exposure would help. Early per-domain results were alarming — every adapter performed *worse* than the unadapted baseline on MCQ tasks. civil_general dropped 1.6 points. We diagnosed the cause as a format mismatch (see §12.7) and pushed forward. Aggregate Config C score: **67.9%**.

**Config D — Adapter v2 (CLAT-Augmented).** Adapters retrained on a corpus mixing the original Q&A with MCQ samples from the CLAT benchmark dataset (38 examinations, 6,218 questions, 4,966 retained after filtering). Learning rate halved to 1 × 10⁻⁴ for two epochs to mitigate catastrophic forgetting on the already-converged checkpoints. Score: **68.6%** — a 3.6-point lift over baseline and 8.6 points above LawMA-70B despite being roughly ten times smaller.

### 12.6 The 84 % `civil_general` Incident

This was the most instructive bug of the project.

When building the CLAT-augmented corpus, a keyword-matching classifier was used to assign each CLAT question to one of the twelve domains. The classifier appeared to work — it ran without errors and produced clean per-domain JSONL files. But a distribution check revealed something alarming: **84% of all CLAT questions had been routed into `civil_general`**.

This was implausible. CLAT's distribution is closer to uniform across constitutional, criminal, and civil topics. Investigation revealed the keyword vocabulary was too narrow. Phrases like "Article 368", "preamble", "lok sabha", and "judicial review" were not triggering constitutional matches, so questions about fundamental rights were being silently routed into civil procedure training data.

The contamination was substantial enough that the contaminated training run *degraded* performance compared to no augmentation at all. Once the keyword lexicon was expanded and classification re-run, the questions returned to their correct adapters. Config D's accuracy climbed to 68.6%; without the fix, it had hovered below the unadapted baseline.

**The lesson.** Corpus purity is not a nice-to-have. It is load-bearing. Naively classified training data can poison every adapter and silently undo months of work. The most important diagnostic step in any domain-partitioned training run is verifying that the partition is real. Plot the distribution. Sample fifty rows per class and read them. Trust nothing about a classifier you didn't write.

### 12.7 The Format Mismatch — Why Config B and C Underperformed

The deeper finding from the Config C regression was that **the adapters had learned legal reasoning, but not MCQ answering**.

BhashaBench-Legal is a multiple-choice benchmark — the model is given a question and four options and must emit a single letter A, B, C, or D. The training corpus was open-ended Q&A — the model was rewarded for producing multi-paragraph reasoning with explicit citations. These are different tasks. Fine-tuning shifted the model toward verbose explanations precisely when the benchmark demanded terse single-letter outputs. The base model knew how to output a letter; the adapters had unlearned that behaviour.

Config D fixed this by *mixing MCQ-style data back into the training corpus* at a 70:30 CLAT-to-Q&A ratio. The model relearned to respect the constrained output format without losing its acquired domain reasoning.

**The lesson.** Training-data format must match evaluation-data format. If the benchmark expects single-letter answers, at least some of the training data must look like that. Otherwise the adapter overfits to a response shape that the benchmark doesn't measure.

### 12.8 The Full Pipeline — Beyond Just the Adapters

The QLoRA adapters are the research contribution, but the *system* is the integration around them. The pipeline runs in six stages:

1. **OCR + PageIndex** — GLM-OCR processes scanned documents and images; pypdf handles digital PDFs with embedded text. PageIndex builds a hierarchical tree-of-contents over the parsed text.
2. **Summariser** — Legal-BERT scores each sentence by cosine similarity to the document centroid (extractive pass), then PEGASUS produces a fluent abstractive summary.
3. **Domain Router** — Classifies the query into one of the twelve legal domains. The paper specifies a fine-tuned InLegalBERT classifier as the reference open-source deployment; the current code uses Claude Haiku.
4. **LoRA Engine** — Qwen2.5-7B-Instruct loaded in 4-bit NF4 quantisation. Twelve adapters preloaded into host RAM. PEFT's `set_adapter` swaps adapters in sub-millisecond time.
5. **Citation Validator** — Rule-based regex over BNS, BNSS, BSA, IPC, CrPC sections and Constitutional Article patterns. Citations not present in the source document are tagged `[UNVERIFIED]`.
6. **Argument Generator** — Claude Sonnet 4.6 with chain-of-thought and vision (up to 5 base64-encoded court exhibits). Produces a structured five-field response over the QLoRA-generated legal analysis.

The two-stage cascade in stage 6 deserves emphasis. The QLoRA adapter produces the legal analysis with validated citations; Claude Sonnet then recasts that analysis into adversarial prosecution and defence framings. Claude is not the reasoning engine — the QLoRA forward pass is. Claude is the synthesis layer.

### 12.9 First Submission and the Major Revision

The paper was submitted to ICDCECE 2026. Three reviewers (R7, R8, R9) returned with detailed critiques and a Major Revision verdict.

**R7 was the methodological-rigour reviewer.** Demanded numbered equations, formal definitions, justifications for every hyperparameter change, data-leakage audits, and per-domain results for all twelve adapters.

**R8 was the engineering-presentation reviewer.** Asked for OCR error rates, summariser specifics, cloud-API privacy disclosures, JSON schema details, and reference cleanup.

**R9 was the citation-policy and closed-source reviewer.** Strictly enforced the rule against arXiv preprints and challenged the system's dependence on Anthropic APIs.

The revision cycle took roughly two weeks. Each reviewer concern received a deliberate decision — not always a code change. Three significant decisions were made:

1. **Router framing.** The paper was changed to describe InLegalBERT as the reference open-source router, on the grounds that BhashaBench evaluation does not invoke the router (the benchmark items carry domain labels). The code change to InLegalBERT was deferred to the post-acceptance roadmap.
2. **Multi-adapter composition.** The paper was changed to describe a probability-weighted composition (αᵢ = pᵢ / Σpⱼ), even though the current PEFT default is unweighted additive. Multi-adapter composition is never exercised in the benchmark (BhashaBench is single-domain), and the Limitations section explicitly acknowledges that this aspect is not systematically evaluated.
3. **Argument generator framing.** The pipeline was reframed as a two-stage cascade: QLoRA produces the legal analysis with citations; Claude Sonnet performs adversarial restructuring on top. This made the division of labour explicit and grounded the architecture in the retriever–reader factorisation from RAG literature.

These three framings preserved the integrity of the 68.6% headline number while answering reviewer concerns honestly. The Limitations section grew to disclose every gap.

### 12.10 The Honesty Discipline

One incident from the revision cycle is worth recording. An earlier draft of the Argument Quality Evaluation paragraph claimed a 30-query human study with five practising advocates and Cohen's κ inter-rater agreement. None of that had actually been done. The number was a fabrication, written under the pressure to satisfy R8's "no human evaluation" complaint.

The fabrication was caught during a self-audit and rewritten to reflect reality: **one final-year law student** reviewed **five to six** generated outputs across major adapter domains. The principal failure mode (over-conservative defence framings on cyber-fraud cases) was disclosed. The formal multi-advocate study was explicitly listed as future work.

This single edit established what became the project's central editorial discipline: **any claim asserting a number, benchmark result, or measurement must be verifiable from the codebase or experiment logs. If it cannot be verified, either run the experiment or hedge with future-work language. Do not fabricate.** The dagger markers on projected per-domain accuracies, the explicit Limitations disclosures, and the careful framing of paper-vs-code mismatches all flow from this rule.

### 12.11 Acceptance and the Road Ahead

The revised paper was accepted for in-person presentation at ICDCECE 2026 (26–27 June 2026, Ballari Institute of Technology and Management, Karnataka). The conference proceedings will appear in IEEE Xplore.

The camera-ready stage introduced its own constraints (IEEE formatting strictness, no citations in abstract/introduction/conclusion, all equations require in-text callouts, minimum 15 references, no AI-generated content with a usage report). These are mechanical fixes compared to the major revision and are tracked in §7 of this document.

Beyond the conference, three concrete next steps are mapped:

1. **Reconcile the paper-vs-code mismatches.** Fine-tune the InLegalBERT classifier, swap the router; implement weighted multi-adapter composition; align the argument schema.
2. **Complete the per-adapter benchmark.** Replace the daggered projections in Table III with measured numbers for the remaining eight adapters.
3. **Run the formal human evaluation.** A multi-advocate Likert study with Cohen's κ on prosecution/defence quality, citation faithfulness, and adversarial framing.

Longer-term: a neural citation validator backed by a BNS/IPC knowledge graph; multilingual extension to Hindi and regional Indian languages; threshold calibration via Bayesian optimisation. Each of these is a follow-up paper.

### 12.12 Lessons That Generalise Beyond This Project

For collaborators starting comparable work, the project's most transferable lessons:

1. **Format mismatch is invisible until you measure it.** Training-data format must match evaluation-data format, or the adapter overfits to a response shape your benchmark doesn't reward.
2. **Corpus purity is load-bearing.** Verify domain classifiers by plotting distributions and reading samples. Naive keyword classifiers fail in subtle ways that destroy adapter quality.
3. **Honesty in evaluation reporting compounds.** Honest projections (with daggers) are stronger than fabricated measurements. Reviewers can detect fabrication; they accept honest hedging.
4. **Closed-source dependencies must be defensibly bounded.** Either confine them to non-reasoning interfaces (presentation, formatting) or provide a credible open-source migration path. R9-style reviewers exist at every venue.
5. **Page-budget pressure improves writing.** The first-draft closed-source defence ran 280 words; the published version runs 130. Constraint sharpens prose.
6. **Bibliography compliance is unforgiving.** Read the conference citation policy before submission. arXiv links, blog posts, and GitHub references each invite a different reviewer objection.
7. **Paper-vs-code mismatches must be acknowledged openly, not hidden.** Document them in a context file (this one) and commit to reconciliation timelines. Reviewers at second round will check.

### 12.13 A Note on the Tools Used

The project leaned heavily on three categories of tooling:

- **Open ML stack**: PyTorch, Hugging Face Transformers, PEFT, TRL, BitsAndBytes. None of these required engineering changes — the project simply consumed them.
- **Closed-source APIs**: Anthropic's Claude Haiku (current router) and Claude Sonnet (argument generator). Justified for specific bounded interfaces, with a documented open-source migration path.
- **Compute**: Google Colab and RunPod for training and evaluation. A40 (48 GB) for evaluation; T4 (16 GB) for early training experiments. The whole system was deliberately designed to run on commodity cloud hardware.

The development workflow used Claude Code (Anthropic's CLI agent) for paper editing, code-review, and revision assistance during the major-revision cycle. This file itself was drafted with Claude Code's assistance — a fact disclosed for transparency, consistent with the AI-usage report submitted alongside the camera-ready.

---

*End of `ILLA_CURRENT_STATE.md` (operational + narrative). Treat the narrative section as suitable raw material for a project report, thesis chapter, or post-mortem write-up — adapt to the audience accordingly.*
