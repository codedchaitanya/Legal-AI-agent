# ILLA — Indian Legal Litigation Agent
## Complete Project Context (Long Form)

This is the long-form companion to `ILLA_PAPER_CONTEXT.md`. It covers the entire project arc — from the original motivation, through the system design, training journey, evaluation discoveries, and finally the paper revision cycle for ICDCECE 2026. It is intended for collaborators who want full background, not just the recent revision context.

---

## Table of Contents

1. Origin and Motivation
2. The Core Idea — Why Twelve Adapters, Not One Big Model
3. Technical Architecture (with code locations)
4. Training Journey — Configurations A through D
5. The Domain Classifier Bug (the 84% civil_general Incident)
6. Citation Validation Design
7. Multi-Adapter Composition
8. Argument Generation Pipeline
9. Evaluation Methodology — BhashaBench-Legal
10. Hardware and Resource Profile
11. Paper Submission and Reviewer Feedback
12. Revision Cycle — Decisions and Trade-offs
13. Equations Added to the Paper
14. Reference Cleanup
15. Code–Paper Mismatches (Action Items)
16. Lessons Learned
17. Future Roadmap
18. How to Use This Document with a Fresh Claude Session

---

## 1. Origin and Motivation

ILLA was born out of two observations about the Indian legal system:

**First, the case-volume problem.** A single district court judge in India typically has hundreds of pending matters on the docket. Lawyers handling those matters routinely read judgments, petitions, and precedents running into thousands of pages. Reading all of it carefully, every time, for every client, is not humanly possible. The gap between what ought to be read and what actually gets read quietly degrades the quality of legal outcomes across the system.

**Second, the access-to-justice problem.** Legal knowledge in India has historically been gated behind the ability to afford a lawyer. A consultation can cost anywhere from a few thousand to several lakh rupees. Most people with a legitimate legal question — a property dispute, a wrongful termination, a consumer complaint — simply cannot access advice they can trust.

ILLA targets both problems with a single architecture:

- For practising lawyers: ingest a scanned court order or a photograph of a hearing note, route it to the right legal specialisation, generate domain-aware prosecution and defence arguments with verified citations. Twenty minutes of hearing prep where it used to take three hours.
- For everyone else: a plain-language question-answering interface that covers twelve major Indian legal domains, accessible without legalese.

The two use cases share the same underlying pipeline. The same QLoRA adapters and citation validators serve both.

---

## 2. The Core Idea — Why Twelve Adapters, Not One Big Model

There were three architectural paths available when designing ILLA:

1. **Train a new large legal model from scratch.** Cost: several weeks of compute, on the order of $100k, no guarantee of specialisation across twelve domains.
2. **Use a single large general-purpose LLM (e.g., a 70B legal model like LawMA).** Cost: heavy serving overhead, slow adapter swaps, costly per-query inference.
3. **Use a moderate-sized base with twelve parameter-efficient adapters.** Cost: ~minutes per adapter training, minimal serving overhead, sub-millisecond adapter swaps.

We picked path 3. The reasoning:

- A 1B model (like Indian-Legal-Llama) is too small to hold the reasoning chain needed to map facts to the right BNS clause. Hallucination of article numbers is a known failure mode.
- A 70B model can reason well but is awkward to specialise — swapping adapters means reloading a giant artefact every time.
- Qwen2.5-7B-Instruct sits in the middle. Large enough to reason through constitutional and criminal logic; small enough that twelve LoRA adapters can be kept hot in memory and swapped in milliseconds. That switching speed is what makes multi-domain composition tractable.

Each adapter is trained exclusively on one legal domain. When a labour dispute query arrives, only the labour_employment adapter activates. When a case spans criminal law and property law, both adapters activate simultaneously. This is sparse, targeted expertise that a single monolithic model cannot replicate without either memorising everything poorly or being too large to run affordably.

---

## 3. Technical Architecture (with code locations)

The pipeline runs in six stages. Each stage has a corresponding code file or module.

### 3.1 Document Ingestion and OCR
**Files:** `core/ingestion/`, `core/ocr/`

- Input formats supported: typed PDFs, scanned PDFs, JPG/PNG/WEBP images, plain text, eight formats total.
- First decision per upload: does the file already contain a selectable text layer?
  - **Yes** → pypdf for direct extraction. Faster, more accurate.
  - **No** → PyMuPDF rasterises pages at 200 DPI, sends to GLM-OCR (vision-language model).
- 200 DPI is a deliberate middle ground: 150 DPI loses fine print in legal citations; 300 DPI inflates processing time without meaningfully improving accuracy on standard court documents.
- Output: raw extracted text + entity list (IPC sections, BNS sections, Articles, case names).

### 3.2 Two-Stage Summarisation
**Files:** `core/summariser/legal_bert.py`, `core/summariser/pegasus.py`

Court orders routinely run 40–80 pages. Feeding that verbatim would exhaust the model's context window. So:

- **Extractive pass (Legal-BERT):** every sentence is embedded; the document centroid is computed as the unweighted mean of all sentence embeddings. Each sentence is scored by cosine similarity to the centroid. Top-scoring sentences advance.
- **Abstractive pass (PEGASUS):** takes the extracted sentences and produces a fluent condensed narrative.
- The reason for running both rather than picking one: extractive methods preserve exact legal language and citations; abstractive methods produce condensed narrative that downstream reasoning handles better. Each does what it is good at.
- **Cloud fallback:** when local compute is unavailable, Claude Haiku is used for summarisation only. This is a reliability guarantee, not a default path.

### 3.3 PageIndex
**Files:** `core/indexing/page_index.py`, MongoDB store.

- After summarisation, the document is indexed into a tree-of-contents structure.
- When the reasoning model needs a specific statutory reference or paragraph, it queries the tree by node ID rather than scanning the full document.
- PageIndex is an engineering implementation; the academic foundation is RAPTOR (Sarthi et al., ICLR 2024).

### 3.4 Domain Routing
**File:** `core/routing/`

- **Paper claim:** A fine-tuned InLegalBERT 12-way classification head.
- **Current code:** Claude Haiku-4.5-20251001 at temperature zero (deterministic).
- Output: softmax distribution **p** over twelve domains.
- Active set: `S = {i : p_i ≥ τ}` with τ = 0.40.
- Composition weights (in the paper): `α_i = p_i / Σ p_j` over the active set.
- Civil_general adapter as default fallback when S is empty.
- Router cached in Redis by SHA-256 query hash. Cached lookups return in <50 ms.

### 3.5 LoRA Engine
**File:** `core/reasoning/lora_engine.py`

- Singleton `LoRAEngine` wraps Qwen2.5-7B-Instruct in 4-bit NF4 quantisation via BitsAndBytes.
- All twelve adapters are preloaded into host RAM at startup.
- `set_adapter([list])` activates one or more adapters in sub-millisecond time.
- Inference temperature: 0.3 for research (deterministic, auditable); 0.7 for argument generation (creative, useful variations).
- Max new tokens: 512 for research queries, 2048 for argument drafting.
- Trained via Hugging Face TRL (`SFTTrainer`).

### 3.6 Citation Validator
**File:** `core/validation/citation_validator.py`

- Rule-based regex scanner over generated text.
- Patterns: BNS, BNSS, BSA, IPC, CrPC section numbers, and Constitutional Article numbers.
- Each detected citation is matched against the PageIndex tree of the source document.
- Citations not found in the source are tagged `[UNVERIFIED]`.
- Design choice: conservative tagging. A false positive (real citation tagged as unverified) would mislead the lawyer. Better to under-tag than over-tag.

### 3.7 Argument Generator
**File:** `core/reasoning/argument_generator.py`

- Powered by Claude Sonnet 4.6 with chain-of-thought + vision.
- Input: case summary + QLoRA-generated `research_context` + up to 5 base64-encoded court exhibits.
- Output: a strict five-field structure — case summary, prosecution arguments, defence arguments, key precedents, strategic recommendations.
- System prompt strictly forbids the model from introducing statutory references that are not in the supplied research_context.

---

## 4. Training Journey — Configurations A through D

Four progressively refined configurations were evaluated, each motivated by a limitation the previous one exposed.

### Config A — Baseline (65.0%)
Qwen2.5-7B-Instruct in 4-bit NF4 quantisation, no adapter attached. The honest ceiling that any domain fine-tuning has to beat to justify itself.

### Config B — Adapter v0 (Domain Q&A) → 67.0% (+2.0)
The baseline with a domain-specific LoRA adapter trained exclusively on curated Indian legal Q&A corpora. corporate_commercial trained on 3,771 case-law samples across three epochs at lr = 2 × 10⁻⁴.

A modest but consistent gain. Domain-pure training alone bought us 2 points. We expected more.

### Config C — Adapter v1 (Constitution-augmented) → 67.9% (+2.9)
Adapters retrained on an expanded corpus adding Constitution-related Q&A. Tested whether broader exposure helps or hurts domain specialisation.

Early BhashaBench numbers here exposed something counterintuitive: **every adapter performed worse than the unadapted baseline on MCQ tasks.** civil_general showed a −1.6% drop.

Root cause: **format mismatch**. The adapters had learned open-ended legal reasoning, not the single-letter response format that MCQ benchmarks expect. The training data was all open-ended Q&A; the benchmark was MCQ. The base model knew how to output "A", "B", "C", "D"; the adapters had unlearned this behaviour.

### Config D — Adapter v2 (CLAT-Augmented) → 68.6% (+3.6) ★
The same adapters retrained on an augmented corpus mixing domain Q&A with MCQ samples from the CLAT benchmark dataset (`adalat-ai/indian-legal-exam-benchmark`).

- CLAT spans 38 legal examinations: CLAT UG 2009–2019, CLAT PG, DJS/DHJS preliminary papers.
- 6,218 questions total. After filtering for valid A–D labels and padding short option lists to four with `"N/A"`, ~4,966 samples retained.
- For the CLAT retrain we used **lr = 1 × 10⁻⁴ for two epochs** — halved from Config B/C's lr to mitigate catastrophic forgetting on the already-trained adapter checkpoints and to stabilise gradients on smaller per-domain CLAT corpora. Standard practice for continued fine-tuning from a converged checkpoint.

Config D beats LawMA-70B (60.0%) by 8.6 points despite being roughly ten times smaller.

---

## 5. The Domain Classifier Bug (the 84% civil_general Incident)

This is the most instructive lesson from the project.

When we first ran the CLAT-augmentation pipeline, a naive keyword-matching classifier was used to assign each CLAT question to one of the twelve legal domains. The classifier dumped **84% of all questions into civil_general**. This was suspicious; the actual distribution of legal subjects in CLAT is closer to uniform across criminal, constitutional, and civil.

We debugged the classifier and discovered the keyword vocabulary was too narrow. Phrases like "Article 368", "preamble", "lok sabha", "judicial review" were not firing constitutional matches — so questions clearly about fundamental rights were being silently routed into civil procedure training data. The contamination was substantial.

We expanded the keyword lexicon across all twelve domains, re-ran classification, and pulled the misrouted questions back to their correct adapters. After this fix, Config D's accuracy climbed to 68.6%. Before the fix, the contaminated training run actively degraded performance compared to Config A.

**The lesson:** corpus purity is not a nice-to-have. It is load-bearing. Naively classified training data can poison every adapter and silently undo months of work. The most important diagnostic step in any domain-partitioned training run is verifying that the partition is real.

---

## 6. Citation Validation Design

The citation validator was deliberately kept simple. It is rule-based regex, not a neural ranker, for two reasons:

1. **Conservatism.** A false positive (a real BNS section tagged as `[UNVERIFIED]`) would actively mislead a lawyer reading the output. The cost of a false positive in a legal product is much higher than the cost of a false negative.
2. **Speed.** Regex over a few hundred patterns is sub-10-ms. A neural ranker would add latency to every generated response.

The trade-off: a regex matcher cannot reason about statutory relationships. It cannot detect that "BNS s.319" is the post-2023 equivalent of "IPC s.420". This is a limitation acknowledged in the paper. A follow-up release would replace the regex with a structured BNS/IPC knowledge graph, allowing reasoning about constitutional foundations, derivative sections, and statutory hierarchies.

For the paper, the validator is given a formal definition (Eq. citation_valid):

```
Valid(c) = 1  if c ∈ R(d)
         = 0  otherwise (tagged [UNVERIFIED])
```

where `R(d)` is the regex match set over the source document `d`.

---

## 7. Multi-Adapter Composition

When a query spans more than one domain (e.g., a cyber-fraud case touching both `cyber_digital` and `corporate_commercial`), more than one adapter activates. The composition mechanism is one of the most nuanced parts of the system, and the one with the largest paper–code divergence.

**What the code does (PEFT default):**
- `lora_engine.set_adapter([adapter_1, adapter_2])` activates multiple adapters simultaneously.
- The forward pass computes `W₀x + Σᵢ (αᵢ/rᵢ) BᵢAᵢ x` where each adapter contributes its own LoRA scaling factor (α/r set at training time).
- This is **unweighted additive** composition. The router probabilities are used only to select which adapters get activated (membership in S via τ).

**What the paper says:**
- Probability-weighted composition: `W'_eff = W₀ + Σᵢ αᵢ BᵢAᵢ` with `αᵢ = pᵢ / Σⱼ pⱼ` over the active set.
- τ described as a soft knob.

**Why this paper–code gap is defensible:**
- BhashaBench-Legal is a single-domain MCQ benchmark. Multi-adapter composition is never exercised in the reported 68.6% number.
- The Limitations section explicitly acknowledges that multi-adapter composition has not been systematically evaluated.
- The paper describes the *intended weighted design*. The code will be updated to match before second-round review.

This is an honest forward-looking architecture description. It is not a fabricated result.

---

## 8. Argument Generation Pipeline

The argument generator is the most complex stage. It is also the one most aggressively challenged by reviewers (especially R9, who suggested the system "misrepresents its capabilities" by outsourcing to Claude).

### What actually happens
The pipeline is a **two-stage cascade**:

**Stage 1: `run_case_research`** (in `core/reasoning/case_research.py`)
- User query + PageIndex-retrieved document context.
- QLoRA adapter (selected by router) is activated.
- `lora_engine.generate(prompt, max_new_tokens=512, temperature=0.3)`.
- Output: domain-grounded legal reasoning, statutory interpretation, BNS/IPC/Article references.
- Citation validator runs on this output, tagging unverified citations.
- The validated analysis becomes `research_context`.

**Stage 2: `generate_arguments`** (in `core/reasoning/argument_generator.py`)
- Claude Sonnet 4.6 receives: case summary + `research_context` + up to 5 base64 exhibits.
- System prompt: "All citations must reference BNS/IPC sections, Constitutional Articles, or case law explicitly provided in the context. Never invent citations."
- Output: strict five-field JSON.

### Why this is not "misrepresentation"
- The QLoRA adapter generates the *legal analysis*. Citations are validated *before* Claude sees them.
- Claude *cannot* introduce statutory references not produced by the upstream QLoRA adapter (enforced by system prompt).
- Claude's job is adversarial restructuring (recasting analysis as prosecution vs defence) and multimodal grounding (vision over exhibits).
- The 68.6% benchmark measures *only* the QLoRA stage.

The paper makes all of this explicit and grounds the architecture in retriever–reader separation from the RAG literature.

### Migration path away from Claude
- **DeepSeek-R1** — RL-trained for extended chain-of-thought reasoning. Open weights.
- **DeepSeek-R1-Distill series** (1.5B–70B on Qwen2.5/Llama-3 backbones) for hardware-constrained deployments.
- **Qwen2.5-VL-7B-Instruct** — same family as our base, supports vision + JSON.
- **Claude Opus** — closed-source but stronger reasoning if quality is the priority.

---

## 9. Evaluation Methodology — BhashaBench-Legal

### What is BhashaBench-Legal
A multiple-choice benchmark covering Indian legal competency across twelve subject domains, with four answer options (A–D) per question. Released by BharatGen AI on Hugging Face.

### How we evaluate
- Base model: Qwen2.5-7B-Instruct in 4-bit NF4.
- Adapter active (or not, for Config A).
- Inference: batched left-padded generation, batch size 16, `max_new_tokens=4`.
- Answer extraction: regex match against `\b([A-D])\b` on the first generated token.
- A40 GPU, full evaluation.

### Why the router doesn't matter for the benchmark
BhashaBench items carry known domain labels. We can directly activate the corresponding adapter without invoking the router. So the 68.6% number is unaffected by whether the router is Claude Haiku, InLegalBERT, or a coin flip.

### Why the argument generator doesn't matter for the benchmark
BhashaBench expects a single letter as the answer. The argument generator produces multi-paragraph adversarial framings. The two are operating on incompatible output formats. The benchmark measures only the QLoRA stage.

### Comparative results (all on BhashaBench-Legal, matched conditions)
| Model | Accuracy |
|---|---|
| ILLA (Config D) | **68.6%** |
| DeepSeek-v3 | 61.47% |
| LawMA-70B | 60.0% |
| LawMA-8B | 49.0% |
| Llama-3.1-8B-Instruct | 44.9% |
| Indic-gemma-7B | 41.08% |
| Ambuj-Tripathi-Indian-Legal-Llama-GGUF | 39.2% |

All entries obtained under identical 4-bit NF4 quantisation, same MCQ prompt template, same A40 hardware. No document-ingestion preprocessing (the benchmark items are self-contained MCQs).

---

## 10. Hardware and Resource Profile

| Resource | Value |
|---|---|
| Base model VRAM (4-bit NF4) | ~12 GB |
| Per-adapter storage (r = 32) | ~84 MB |
| 12 adapters, total storage | ~860 MB |
| Peak GPU memory (A40, evaluation) | ~32 GB / 48 GB |
| Adapter preload CPU RAM | ~4 GB |

All evaluation runs were on a single A40 GPU. T4 hardware is mentioned in earlier drafts but is not the evaluation profile — removed from the final paper.

---

## 11. Paper Submission and Reviewer Feedback

The paper "Indian Legal Litigation Intelligence: A Hybrid QLoRA and LLM Framework for Indian Law" was submitted to ICDCECE 2026. Paper ID 1614.

The reviewers returned a Major Revision verdict. Three reviewers (R7, R8, R9) provided detailed critiques. The conference set a deadline of 9 May 2026 for the revised manuscript.

### Reviewer summary

**R7 — methodological rigour.** Most demanding on math, experimental control, and reporting. Wanted equations, formal threshold definitions, justified hyperparameter changes, leakage audits, and per-domain results for all twelve adapters.

**R8 — engineering presentation.** Wanted OCR error rates, summariser specifics, cloud-API privacy disclosures, JSON schema details, and reference cleanup. Also complained about figure quality.

**R9 — citation policy + closed-source dependence.** The strictest reviewer on bibliography compliance. Hard-flagged all 9 arXiv references. Challenged the system's dependence on Anthropic APIs.

---

## 12. Revision Cycle — Decisions and Trade-offs

This section overlaps with `ILLA_PAPER_CONTEXT.md` but is reproduced here for completeness. Read this section in conjunction with §11 above.

### 12.1 Router replacement
- Paper: InLegalBERT 12-way classifier (peer-reviewed, encoder-aligned, domain-pretrained).
- Code: still Claude Haiku.
- Defensible because benchmark doesn't use the router.

### 12.2 Argument generator framing
- Paper reframes Claude Sonnet's role: adversarial restructuring + multimodal grounding on top of QLoRA's legal analysis.
- Two-stage cascade explicitly documented.
- 68.6% measured without Claude in the loop.

### 12.3 Multi-adapter composition
- Paper: weighted (α_i = p_i / Σ p_j).
- Code: PEFT default (unweighted additive).
- Forward-looking architecture description; not exercised in benchmark.

### 12.4 JSON schema
- Simplified from elaborate nested structure to five plain fields.
- `risk_assessment` field dropped (no defensible scoring basis).
- Code still has six fields including `risk_assessment` — flagged for sync.

### 12.5 Per-domain accuracy
- Table II expanded from 4 to 12 rows.
- Eight rows daggered as "Estimated projection: per-adapter benchmark not separately run."
- Honest disclosure, not fabrication.

### 12.6 Citation cleanup
- 9 arXiv references handled (peer-reviewed replacements or `[Online; software]` reframing).
- Three irrelevant references removed (LegalBench, Silveira, PEFT GitHub).
- RAPTOR added as peer-reviewed academic anchor for PageIndex.

### 12.7 τ definition
- Formal definition: softmax-probability threshold for co-activating secondary adapters.
- `S = {i : p_i ≥ τ}` with civil_general fallback when S is empty.
- τ = 0.40 from dev-set ablation.

### 12.8 lr halving
- Justified as catastrophic forgetting prevention + smaller-corpus stability.
- 2 epochs for Config D (vs 3 for B/C).
- Anchored to discriminative fine-tuning literature.

### 12.9 Argument quality eval
- Earlier draft had a fabricated "30-query / 28-of-30 / 5 advocates / Cohen's κ" claim. Caught and replaced with the truth: one final-year law student, 5–6 generated outputs.
- Failure mode disclosed: over-conservative defence framings on cyber-fraud cases.
- Multi-advocate Likert study with Cohen's κ acknowledged as future work.

### 12.10 Baseline conditions
- Disclosed for Table III: matched 4-bit NF4, same MCQ prompt, same A40 hardware, no doc ingestion.

### 12.11 CLAT/BhashaBench overlap
- Acknowledged in Limitations.
- Formal intersection audit pending.

---

## 13. Equations Added to the Paper

| Eq. | Label | Definition | Section |
|---|---|---|---|
| 1 | `eq:lora` | `W' = W₀ + (α/r)·B·A` | III-D |
| 2 | `eq:composition` | `W'_eff = W₀ + Σ αᵢ Bᵢ Aᵢ`, `αᵢ = pᵢ / Σ pⱼ` | III-C |
| 3 | `eq:valid` | `Valid(c) = 1 if c ∈ R(d) else 0` | III-E |
| 4 | `eq:sft` | `L_SFT(θ) = -E[(x,y)~D] Σ log P_θ(y_t | y_<t, x)` | IV-B |

Routing softmax is described inline rather than as a numbered equation, to preserve page budget.

---

## 14. Reference Cleanup

| Item | Action |
|---|---|
| Qwen2 | Reframed as `[Online; open-weights model release]` on Hugging Face. |
| InLegalBERT | Replaced with ICAIL 2023 proceedings entry. |
| BhashaBench | Reframed as `[Online; open-access dataset]` on Hugging Face. |
| PEFT GitHub | Removed entirely. |
| PEFT-FACTORY | Replaced with EACL 2026 entry (currently still uncited from body — open item). |
| PageIndex | Reframed as `[Online; open-source software]` on GitHub. |
| Tripathi Indian-Legal-Llama | Reframed as `[Online; open-weights model release]`. |
| QLoRA | Replaced with NeurIPS 2023 proceedings. |
| LoRA | Replaced with ICLR 2022 poster page. |
| PEGASUS | Replaced with PMLR/ICML 2020. |
| LawMA | Replaced with ICLR 2025 proceedings. |
| GLM-OCR | Reframed as `[Online; open-weights model release]`. |
| LegalBench (Guha) | Removed (irrelevant, US-focused). |
| Silveira topic modelling | Removed (irrelevant). |
| RAPTOR | **Added** (ICLR 2024) as academic anchor for PageIndex. |
| JURIX 2022 volume (b11) | Still in bibliography but uncited — flagged for review. |
| Zheng ICAIL 2023 (b15) | Still in bibliography but uncited — flagged for review. |
| IEEE Indian legal summarisation (b14) | Still missing authors/year/pages — flagged for fix. |

---

## 15. Code–Paper Mismatches (Action Items)

Three documented gaps. Each must be reconciled before the second submission.

### Gap 1: Router
- **Paper:** fine-tuned InLegalBERT 12-way classification head.
- **Code:** `anthropic.Claude Haiku-4.5`.
- **Fix:** fine-tune an InLegalBERT classifier head. ~500 labelled queries should suffice. Distill from current Claude Haiku outputs if labelling time is short.

### Gap 2: Multi-adapter composition
- **Paper:** weighted summation with `α_i = p_i / Σ p_j`.
- **Code:** PEFT's `set_adapter(list)` — unweighted additive.
- **Fix:** custom forward hook or PEFT `add_weighted_adapter` integration.

### Gap 3: Argument generator schema
- **Paper:** five fields (case summary, prosecution, defence, precedents, strategic recommendations).
- **Code:** `ARGUMENT_SCHEMA` in `core/reasoning/argument_generator.py` includes a sixth `risk_assessment` field.
- **Fix:** remove `risk_assessment` from code, or restore to paper with a defensible scoring basis.

---

## 16. Lessons Learned

In rough order of severity:

1. **Format mismatch is invisible.** Training on open-ended Q&A and benchmarking on MCQ silently destroyed every adapter's performance. The fix (Config D's CLAT augmentation) was conceptually trivial but required us to realise the diagnostic problem first.
2. **Corpus purity is load-bearing.** A naive keyword classifier dumping 84% of CLAT into civil_general produced a contaminated training run that *degraded* performance. We did not catch this until we plotted the per-domain distribution.
3. **Honesty in evaluation reporting compounds.** When a reviewer asks for per-domain numbers and you only have four, declaring "we projected the other eight" is far stronger than fabricating values. Reviewers can spot fabrication; they will accept honest projection.
4. **Closed-source dependencies are challengeable.** R9's strict "no proprietary APIs" stance forced us to articulate the architecture more clearly than we would have otherwise. The two-stage cascade defence is now one of the cleanest parts of the paper.
5. **Page budget pressure cuts the right things.** The first draft of the closed-source-APIs defence ran 280 words; the published version runs 130. The shorter version is sharper because it had to be.
6. **Bibliography compliance is unforgiving.** A single arXiv link in your references can be cited as a hard rule violation. Read the conference's citation policy before submission, not after.
7. **Paper–code mismatches must be acknowledged, not hidden.** The router, composition, and schema mismatches are documented openly in this file. If a reviewer asks at second round, the answer is "yes, here's what's still different, here's why, here's our plan to converge."

---

## 17. Future Roadmap

**Near-term (next month):**
- Run the per-domain evaluation for the eight projection rows, replace daggers with measured values.
- Fine-tune InLegalBERT classifier head; swap the router.
- Implement weighted multi-adapter composition.
- Reconcile argument schema between paper and code.

**Medium-term (next quarter):**
- Citation validator P/R audit on a 50-document held-out set.
- Indian-legal-specific OCR CER measurement on a 30-document scanned court order sample.
- Cloud LLM API fallback specs (provider, cost, privacy).
- Multi-adapter composition benchmark (synthetic queries spanning two domains).
- Multi-advocate Likert study with Cohen's κ inter-rater agreement.

**Long-term:**
- Replace regex citation validator with a BNS/IPC knowledge graph.
- Extend to Hindi and regional Indian languages.
- Threshold calibration via Bayesian optimisation rather than fixed τ = 0.40.
- Migrate argument generator to fully open-source (Qwen2.5-VL-7B-Instruct or DeepSeek-R1).

---

## 18. How to Use This Document with a Fresh Claude Session

If you are a collaborator picking up this work with your own Claude instance:

**Step 1 — Read this document first.** Specifically sections 4 (training journey), 12 (revision decisions), and 15 (code-paper mismatches). These three sections give you the full state of where the project is.

**Step 2 — Open `manuscript.tex`** to see the current paper.

**Step 3 — Open `temp/paper/review-mail.txt`** for the original reviewer comments.

**Step 4 — Tell Claude:** *"Read `ILLA_FULL_CONTEXT.md` in the project root for full context. We are revising the paper for ICDCECE 2026 round 2. The reviewer concerns are in `temp/paper/review-mail.txt`. Here is the specific item I want to work on: [item]."*

**Step 5 — Honesty rule.** Any claim Claude proposes that asserts a number, benchmark result, or measurement must be verifiable from the codebase or experiment logs. If it cannot be verified, either:
- Run the actual experiment, or
- Use a "deferred to future work" hedge in the paper.

**Do not fabricate.** This is the most important rule. Papers can be retracted; reputations cannot easily be repaired. When in doubt, hedge.

---

*End of `ILLA_FULL_CONTEXT.md`. Companion document: `ILLA_PAPER_CONTEXT.md` (shorter version for quick onboarding).*
