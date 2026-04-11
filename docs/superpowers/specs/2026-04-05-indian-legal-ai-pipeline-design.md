# Indian Legal AI Assistant Pipeline — Design Spec
**Date:** 2026-04-05
**Authors:** Design session with AI assistant
**Status:** Approved, ready for implementation planning

---

## 1. Overview

A hybrid legal AI pipeline that helps Indian lawyers with:
1. **Case Research** (Priority 1) — find relevant precedents, applicable sections (BNS/BNSS/Constitution), and similar judgments
2. **Document Q&A** (Priority 2) — understand client-provided FIRs, contracts, affidavits, charge sheets in plain language
3. **Argument Generation** (Priority 3) — CoT-driven argument drafting with agentic tool calls, critic scoring, and top-K filtering

**Deployment:** Hybrid — cloud LLM APIs (GPT-4.1 / Claude) for generation + self-hosted small models with QLoRA adapters for domain reasoning. Designed for clean upgrade path to fully on-prem.

**Interface:** FastAPI backend + Gradio frontend. Conversation history maintained per `case_id`.

---

## 2. Full Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                         │
│  Client Docs (FIR / contract / affidavit / chargesheet)     │
│       ↓                                                     │
│  [Extractive Summarizer]                                    │
│       ↓ summary + keywords + section refs + doc type        │
│  [Domain Router — Multi-label Classifier]                   │
│       ↓ top-K domain labels + confidence scores             │
│  [PageIndex Builder — Client]  (cached per case_id)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  RETRIEVAL LAYER                             │
│                                                             │
│  PageIndex — Client Docs        PageIndex — Legal DB        │
│  (per-case, built on upload)    (pre-computed, static)      │
│              ↓                           ↓                  │
│         Merged context (relevant pages from both)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 REASONING LAYER                             │
│                                                             │
│  Base LLM (Qwen2.5-7B or Llama-3.1-8B)                     │
│    + active LoRA adapter(s) selected by router              │
│    + conversation history in context                        │
│                                                             │
│  Case Research / Doc Q&A → answered here                    │
│                                                             │
│  Argument Generation:                                       │
│    Stage 1: Domain LoRA extracts legal basis (small model)  │
│    Stage 2: Big Model API (GPT-4.1/Claude) with:            │
│      - CoT: facts → legal basis → precedents → argument     │
│      - tool: fetch_legal_db(node_id)                        │
│      - tool: fetch_case_docs(doc_id, page)                  │
│      - tool: resolve_cross_ref(citation)                    │
│      - tool: search_precedents(query)                       │
│      Agentic: model can re-invoke tools mid-CoT             │
│    Generates K=3-5 argument candidates                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              CRITIC LAYER (argument mode only)              │
│                                                             │
│  Scores each argument on:                                   │
│    Legal Accuracy (0-1)     weight: 0.35                    │
│    Logical Coherence (0-1)  weight: 0.25                    │
│    Evidence Grounding (0-1) weight: 0.25                    │
│    Case Relevance (0-1)     weight: 0.15                    │
│  Threshold: 0.72 (tunable). Return highest-scoring.         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               INTERFACE LAYER                               │
│  FastAPI backend  +  Gradio frontend                        │
│  Conversation history per session / case_id                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Extractive Summarizer

**Input:** Raw client document (PDF / image / text)
**Output:**
- Extractive summary (top sentences, not paraphrased)
- Named entities: people, dates, locations, amounts
- Legal sections mentioned (e.g., "Section 420 IPC" → maps to BNS 318)
- Document type classification: FIR / contract / affidavit / judgment / chargesheet

**Model options (in order of preference):**
1. LLM prompt with structured output (fastest to ship)
2. Legal-BERT / InLegalBERT for NER + classification
3. BERT-based extractive summarizer (e.g., BertSum)

**TODO:** Decide final model after benchmarking on ILDC corpus.

---

### 3.2 Domain Router

**Input:** Summarizer output (keywords, section refs, doc type)
**Output:** Multi-label domain classification with confidence scores

**11 Legal Domains and their LoRA adapters:**

| Domain | Key Laws | Adapter Name |
|--------|----------|-------------|
| Criminal | BNS, BNSS, BSA | `lora-criminal` |
| Constitutional | Constitution of India, landmark SC cases | `lora-constitutional` |
| Civil Procedure | CPC, Limitation Act, Specific Relief Act | `lora-civil` |
| Corporate/Commercial | Companies Act, Contract Act, NI Act, SEBI | `lora-corporate` |
| Family/Personal | Hindu Marriage Act, Muslim Personal Law, POCSO, DV Act | `lora-family` |
| Property/Real Estate | TPA, Registration Act, RERA, Land Acquisition | `lora-property` |
| Labour | Industrial Disputes Act, 4 Labour Codes | `lora-labour` |
| Tax | Income Tax Act, GST, Customs | `lora-tax` |
| IP | Patents Act, Copyright Act, Trademarks Act | `lora-ip` |
| Banking/Insolvency | RBI Act, SARFAESI, IBC, Banking Regulation Act | `lora-banking` |
| Cyber/Data | IT Act 2000, DPDP Act 2023 | `lora-cyber` |

**Special adapters:**
- `lora-argument-gen` — trained on written legal briefs and pleadings (task adapter, not domain)
- `lora-critic` — reward model for argument scoring

**Router implementation phases:**
- Phase 1 (MVP): Zero-shot big model classification (1-2s, ~$0.001/query)
- Phase 2: Fine-tuned BERT classifier on labelled case data (~50ms, free at inference)

**Multi-adapter composition with PEFT:**
```python
model = PeftModel.from_pretrained(base, "adapters/criminal", adapter_name="criminal")
model.load_adapter("adapters/constitutional", adapter_name="constitutional")
# At query time:
model.set_adapter(["criminal", "constitutional"])  # both active simultaneously
```

A single case can activate multiple adapters. Confidence threshold for secondary adapter activation: > 0.40.

---

### 3.3 PageIndex — Two Instances

#### Legal DB PageIndex (Pre-computed, Static)

Built once. Updated only on law amendments.

**Node schema:**
```json
{
  "node_id": "BNS-103",
  "title": "Section 103 — Murder",
  "act": "Bharatiya Nyaya Sanhita 2023",
  "section_number": "103",
  "old_equivalent": "IPC Section 302",
  "summary": "Whoever commits murder shall be punished with death or imprisonment for life...",
  "keywords": ["murder", "culpable homicide", "death penalty"],
  "domain": "criminal",
  "sub_nodes": [],
  "page_range": [88, 89]
}
```

- `old_equivalent` field: maps BNS/BNSS sections to their IPC/CrPC equivalents. Lawyers think in old section numbers — the system silently resolves.
- Pre-computation: parse all bare acts → generate LLM summaries per section (one-time GPT-4.1 cost) → build tree → store in MongoDB
- Estimated corpus: ~50,000 nodes across all 11 domains
- One-time compute: ~4-6 hours. Then static.

#### Client Docs PageIndex (Per-case, Dynamic)

Built on document upload. Cached by `case_id`. Invalidated on new document upload.

**Node schema:**
```json
{
  "case_id": "CASE-2024-001",
  "doc_type": "FIR",
  "node_id": "CASE-2024-001-FIR-P3",
  "title": "FIR Page 3 — Statement of Complainant",
  "summary": "Complainant states accused entered premises at 11pm...",
  "page": 3,
  "sub_nodes": []
}
```

---

### 3.4 LoRA Adapter Training

**Base models to evaluate:**
- Qwen2.5-7B-Instruct (IIT Patna baseline exists)
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3

**Training method:** QLoRA (4-bit quantization + LoRA) via PEFT + TRL

**Training estimates (at ~2 hrs/adapter on A100 40GB):**

| Adapter | Dataset Size | Est. Training Time |
|---------|-------------|-------------------|
| Criminal | 5,000 pairs | 2 hrs |
| Constitutional | 3,000 pairs | 2 hrs |
| Civil Procedure | 3,000 pairs | 2 hrs |
| Corporate/Commercial | 4,000 pairs | 2 hrs |
| Family/Personal | 3,000 pairs | 2 hrs |
| Property/Real Estate | 2,500 pairs | 2 hrs |
| Labour | 2,000 pairs | 2 hrs |
| Tax | 3,000 pairs | 2 hrs |
| IP | 1,500 pairs | 2 hrs |
| Banking/Insolvency | 2,500 pairs | 2 hrs |
| Cyber/Data | 1,500 pairs | 2 hrs |
| Argument Generation | 3,000 briefs | 3 hrs |
| Critic/Reward Model | 2,000 scored pairs | 4 hrs |
| **Total** | **~38,000 samples** | **~29 hrs compute** |

With 4 GPUs in parallel: **~8-10 hrs wall clock time**
Data preparation is the true bottleneck: **3-6 weeks**

**Data sources:**
- Indian Kanoon (scraped judgments)
- InLegalBench dataset
- ILDC (Indian Legal Document Corpus)
- Bare acts (official government PDFs)
- CUAD (for commercial adapter)
- Human-curated Q&A pairs (ongoing, from lawyer feedback)

**IIT Patna Qwen-7B QLoRA assessment:**
- Use immediately as `criminal_v0` baseline adapter
- Weaknesses: 1,500 training pairs only, covers old IPC not BNS, no benchmark eval
- Action: Run InLegalBench against it. Your trained adapters must beat its score.

**Testing multiple base models:**
Train the Criminal adapter on all 3 base models. Evaluate on InLegalBench + domain eval set. Pick best-performing base for all other adapters.

---

### 3.5 Argument Generation — CoT + Agentic Big Model

**Two-stage process:**

Stage 1 — Domain LoRA prepares structured legal brief (small model, fast):
```
case facts + PageIndex merged context
        ↓
small model + domain LoRA(s)
        ↓
structured brief: {applicable_sections, relevant_precedents, key_facts, weak_points}
```

Stage 2 — Big Model CoT with agentic tool calls:
```
System: You are a senior Indian advocate. Reason step by step.

Step 1 — Facts:         Establish what is agreed and what is disputed
Step 2 — Legal Basis:   Which sections apply? → [tool: fetch_legal_db if verification needed]
Step 3 — Precedents:    Find similar judgments → [tool: search_precedents]
Step 4 — Construction:  Build the argument chain
Step 5 — Counter:       Anticipate opposing arguments
Step 6 — Conclusion:    Final argument statement
```

Agentic rule: at any step, if model needs to verify or cross-reference, it fires a tool call and continues CoT with the result. No fixed limit on tool calls per generation.

Generates K=5 argument candidates (default, tunable 3-5). All passed to Critic.

**PEFT in argument generation:**
`lora-argument-gen` adapter (task adapter, not domain) is composed with the active domain adapter:
```python
model.set_adapter(["criminal", "argument-gen"])
# domain adapter (lora-criminal): knows what law says
# task adapter (lora-argument-gen): knows how to write legal arguments
```

---

### 3.6 Critic Model

**Scoring rubric:**

| Dimension | Weight | What it checks |
|-----------|--------|---------------|
| Legal Accuracy | 0.35 | Cited sections exist and are correctly interpreted |
| Logical Coherence | 0.25 | Premise → conclusion chain holds |
| Evidence Grounding | 0.25 | Every factual claim has a PageIndex source |
| Case Relevance | 0.15 | Argument addresses actual case facts, not generic law |

`final_score = 0.35×accuracy + 0.25×coherence + 0.25×grounding + 0.15×relevance`

Default threshold: **0.72** (tunable per deployment).
Return: highest-scoring argument above threshold. If none pass, regenerate at temperature=0.9 (vs default 0.7), up to 2 retries, then flag to lawyer for manual review.

**Implementation phases:**
- Phase 1 (MVP): GPT-4.1 as judge with structured rubric prompt (~$0.02/eval)
- Phase 2: Fine-tuned reward model using human lawyer ratings collected from Phase 1 usage

---

## 4. Quantitative Benchmarks

| Component | Metric | Target |
|-----------|--------|--------|
| Summarizer | ROUGE-L vs human summary | > 0.45 |
| Summarizer | Section extraction F1 | > 0.88 |
| Router | Multi-label F1 per domain | > 0.85 |
| Router | Top-1 accuracy | > 0.90 |
| LoRA Adapters | InLegalBench accuracy | > 65% (baseline IIT Patna: ~45%) |
| LoRA Adapters | LegalBench ISSUE subset | > 60% |
| LoRA Adapters | BERTScore on Q&A | > 0.82 |
| Case Research | Recall@3 | > 0.75 |
| Case Research | MRR | > 0.70 |
| Case Research | NDCG@5 | > 0.72 |
| Argument Gen | Human coherence score (1-5) | > 3.8 avg |
| Argument Gen | BERTScore vs expert arguments | > 0.80 |
| Critic Model | Pearson correlation with human rating | > 0.78 |
| Critic Model | Precision of good-argument filter | > 0.85 |
| Full Pipeline | End-to-end InLegalBench | > 60% |
| Full Pipeline | Case research latency | < 5 seconds |
| Full Pipeline | API cost per argument generation | < $0.08 |

**Benchmark datasets:**
- **InLegalBench** — primary Indian legal Q&A benchmark
- **LegalBench** — general legal reasoning (secondary)
- **ILDC** — Indian Legal Document Corpus (summarizer training/eval)
- **CUAD** — contract understanding (commercial adapter)
- **Custom domain eval sets** — 100 expert-verified Q&A pairs per domain (built incrementally)

---

## 5. Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI |
| Frontend | Gradio |
| LLM fine-tuning | PEFT (QLoRA) + TRL + Transformers |
| Base models | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct |
| Cloud LLM API | GPT-4.1 / Claude (argument generation + critic phase 1) |
| Vector store | None (vectorless — PageIndex only) |
| Document store | MongoDB (legal DB index + case indexes) |
| Embeddings | None required for core pipeline |
| Training hardware | A100 40GB (or 4× T4 for parallel adapter training) |

---

## 6. Pain Points / Open TODOs

### Data
- [ ] **TODO-D1:** Source and clean 38,000 legal Q&A pairs across 11 domains. Indian Kanoon scraping + human curation. Biggest bottleneck.
- [ ] **TODO-D2:** Build IPC → BNS / CrPC → BNSS / Evidence Act → BSA section mapping table. Required for `old_equivalent` field.
- [ ] **TODO-D3:** Curate 3,000 high-quality legal brief samples for `lora-argument-gen` training. Needs practising lawyer review.
- [ ] **TODO-D4:** Collect 2,000 human-rated argument pairs for critic reward model (Phase 2). Requires live app usage first.

### Modelling
- [ ] **TODO-M1:** Benchmark IIT Patna Qwen-7B on InLegalBench as baseline before any training.
- [ ] **TODO-M2:** Compare 3 base models (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B) on Criminal adapter before committing to one base for all domains.
- [ ] **TODO-M3:** Evaluate PEFT `set_adapter()` multi-adapter composition quality vs single adapter. Measure if two adapters active simultaneously degrades accuracy.
- [ ] **TODO-M4:** Define threshold tuning strategy for critic model (start at 0.72, A/B test with lawyers).

### Infrastructure
- [ ] **TODO-I1:** Design pre-computation pipeline for Legal DB PageIndex (~50,000 nodes). One-time GPT-4.1 cost estimate needed.
- [ ] **TODO-I2:** Define cache invalidation strategy for client PageIndex on new document upload.
- [ ] **TODO-I3:** Design fallback when no argument passes critic threshold (regenerate with higher temperature vs lower threshold vs flag to lawyer).
- [ ] **TODO-I4:** Define conversation history truncation strategy (context window management for long sessions).

### Legal/Domain
- [ ] **TODO-L1:** Get legal expert review of domain adapter boundaries. Some cases (e.g., corporate fraud) span Criminal + Corporate + Banking — validate router handles this correctly.
- [ ] **TODO-L2:** Verify all BNS/BNSS/BSA section numbers and their IPC/CrPC/Evidence Act equivalents before building Legal DB index.
- [ ] **TODO-L3:** Define disclaimer/warning system. System must never present itself as replacing a licensed advocate. Required for legal compliance.

---

## 7. Phased Delivery

### Phase 1 — Foundation (Weeks 1-4)
- Legal DB PageIndex pre-computation
- Client doc ingestion + PageIndex builder
- Summarizer + zero-shot router (big model)
- Basic Q&A with IIT Patna adapter as `criminal_v0`
- FastAPI + Gradio MVP UI
- InLegalBench baseline measurement

### Phase 2 — Domain Adapters (Weeks 5-12)
- Train all 11 domain LoRA adapters
- Fine-tune BERT router classifier
- Multi-adapter composition via PEFT
- Full case research pipeline with cross-reference resolution
- Benchmark all adapters, compare base models

### Phase 3 — Argument Generation (Weeks 13-20)
- `lora-argument-gen` adapter training
- CoT + agentic tool call pipeline (big model)
- Critic model Phase 1 (big model as judge)
- Top-K filtering and threshold tuning
- Human lawyer evaluation sessions

### Phase 4 — Hardening (Weeks 21+)
- Critic reward model Phase 2 (trained on collected ratings)
- Fine-tuned BERT router replacing zero-shot
- On-prem option: swap cloud API for local 70B model
- Performance optimisation, latency tuning
- Security, data privacy audit

---

## 8. LoRA Technical Depth — Capabilities, Limits, and Trade-offs

### 8.1 What LoRA Is and Isn't Doing Here

LoRA injects small trainable low-rank matrices into the attention layers (Q, K, V, O projections) of the base model. With rank r=64 on a 7B model, this adds roughly 20–40MB of new weights — approximately 0.1–0.5% of total parameters.

**Critical distinction for this pipeline:**

```
LoRA = Legal Intelligence          PageIndex = Legal Knowledge
──────────────────────────         ──────────────────────────
How to think legally               What the law actually says
How to write legal arguments       Exact BNS/BNSS section text
Interpret a lawyer's query         Real case judgment content
Understand domain terminology      Find which sections apply
Classify case domain (router)      Retrieve Section 103 BNS verbatim
Argument structure and tone        Cross-reference "See Appendix G"
```

**Neither works alone:**
- LoRA without PageIndex = legally fluent model that hallucinates section numbers and case names
- PageIndex without LoRA = precise retriever attached to a model that reasons generically, not legally

The design is sound because LoRA is never asked to be the knowledge store. PageIndex is the ground truth. LoRA shapes how the model reasons over what PageIndex retrieves.

**Fundamental limit:** You cannot memorise 50,000 legal sections into 40MB of LoRA weights. Attempting this causes confident hallucination — the worst possible failure mode for a legal tool. All section content, case citations, and statutory text must come from PageIndex, not from model memory.

---

### 8.2 LoRA Hyperparameter Trade-offs

#### Rank (r) — the most important knob

| Rank | Params added | Use case | Risk |
|------|-------------|----------|------|
| r=8 | ~5MB | Simple style/format adaptation | Underfits complex tasks |
| r=16 | ~10MB | Standard domain Q&A | Good default starting point |
| r=32 | ~20MB | Domain reasoning + terminology | Sweet spot for most adapters |
| r=64 | ~40MB | Complex multi-step tasks | Slight overfitting risk on small data |
| r=128 | ~80MB | Argument generation style | Use only for argument-gen adapter |
| r=256 | ~160MB | Near full-FT expressiveness | Expensive, rarely needed |

**Recommendation per adapter type:**
- Domain Q&A adapters (criminal, constitutional, etc.): **r=32**
- Router/classifier adapter: **r=16** (simpler task)
- Argument generation adapter: **r=128** (legal writing is complex and stylistically demanding)
- Critic/reward adapter: **r=64** (scoring requires nuanced judgment)

#### Alpha (lora_alpha) — controls learning rate scaling

Typically set to `alpha = 2×r`. So r=32 → alpha=64. This is a stable default. Only tune if training loss diverges or underfits badly.

#### Dropout (lora_dropout)

- Small datasets (< 2,000 pairs): use 0.1 to prevent overfitting
- Medium datasets (2,000–5,000 pairs): use 0.05
- Large datasets (> 5,000 pairs): use 0.0 (no dropout needed)

#### Target modules

Standard: `["q_proj", "v_proj"]` (query and value projections only) — fast, good baseline.
Better: `["q_proj", "k_proj", "v_proj", "o_proj"]` — all attention projections, ~2× params, measurably better for reasoning tasks.
Best: Add `["gate_proj", "up_proj", "down_proj"]` (MLP layers too) — best quality, ~3× params. Recommended for argument-gen adapter.

---

### 8.3 Multi-Adapter Composition Trade-offs

PEFT's `set_adapter(["criminal", "constitutional"])` activates multiple adapters simultaneously. Each adapter's delta weights are added to the base model weights independently.

**Trade-off table:**

| Scenario | Expected quality | Risk |
|----------|-----------------|------|
| Single domain adapter | Baseline (100%) | None |
| Domain + task adapter (criminal + argument-gen) | ~97% | Low — complementary tasks |
| Two domain adapters (criminal + constitutional) | ~92–97% | Medium — possible interference |
| Three adapters active | ~85–92% | High — degrades noticeably |

**Why interference happens:** If Criminal adapter was trained with "Section 302 IPC = murder" and Constitutional adapter was trained with unrelated content, the combined delta weights can partially cancel each other in shared attention heads.

**Mitigation strategies:**

Option A — Test and merge: If two adapters are always used together (e.g., criminal cases almost always need constitutional context), pre-merge their weights offline using `add_weighted_adapter()`. One merged adapter, no runtime interference. Recommended for top-3 most common domain combinations.

Option B — Sequential generation: Run domain adapter first → extract structured brief → switch to task adapter for final output. Zero interference, slight latency cost (~200ms extra).

Option C — Accept the degradation: If degradation is < 5% on InLegalBench, composition is fine. Only intervene if measured degradation exceeds threshold.

**TODO-M3 specifically tests this.** Do not assume composition works — measure it.

---

### 8.4 PEFT Method Comparison — Why QLoRA Over Alternatives

| Method | Params trained | Memory | Quality | Training speed | Notes |
|--------|---------------|--------|---------|---------------|-------|
| Full fine-tuning | 100% | ~28GB (7B in bf16) | Best | Slowest | Destroys generalization, impractical for 11 adapters |
| LoRA (fp16) | 0.1–0.5% | ~14GB | Very good | Fast | Standard choice |
| QLoRA (4-bit + LoRA) | 0.1–0.5% | ~6GB | Good (~1–2% below LoRA) | Fast | **Chosen approach** — fits T4 16GB |
| IA³ | 0.01% | ~5GB | Moderate | Fastest | Too light for complex legal reasoning |
| Prompt tuning | 0.001% | ~5GB | Weakest | Fastest | Only for simple style changes, not reasoning |
| Prefix tuning | 0.1% | ~6GB | Moderate | Fast | Less stable than LoRA on instruction-tuned models |

**QLoRA is the right choice for this pipeline** because:
1. Fits on a single T4 (16GB) — the training hardware you have
2. Quality loss vs full LoRA is ~1–2%, acceptable
3. Same PEFT API — all adapters load identically at inference
4. Enables running 4 training jobs in parallel on 4× T4s

**The one exception:** If base model comparison (TODO-M2) shows one model significantly outperforms others, consider training the argument-gen adapter in full LoRA (fp16) for maximum quality on that critical task. Budget ~28GB GPU for that run only.

---

### 8.5 Base Model Selection Trade-offs

Train Criminal adapter on all three base models first. Run InLegalBench. Pick winner for all other adapters.

| Model | Strengths | Weaknesses | Legal reasoning prior |
|-------|-----------|-----------|----------------------|
| Qwen2.5-7B-Instruct | Strong instruction following, multilingual (handles Hindi legal terms), IIT Patna adapter exists as baseline | Less community fine-tune tooling | Moderate — likely seen some Indian legal text |
| Llama-3.1-8B-Instruct | Largest community, most fine-tune resources, strong reasoning | Monolingual bias | Moderate |
| Mistral-7B-Instruct-v0.3 | Very fast inference, efficient architecture | Slightly weaker on long context | Lower — less legal training data likely |

**Recommendation:** Qwen2.5-7B-Instruct is the likely winner due to multilingual ability (Hindi/English code-switching in Indian legal documents is common) and the existing IIT Patna baseline. But measure before committing.

---

### 8.6 Baseline-First Strategy (Critical)

**Before training a single LoRA adapter:**

Run this experiment:
```
PageIndex (legal DB) + base model (no LoRA) + system prompt:
"You are an expert Indian legal assistant. Answer based only on 
the provided legal context. Cite section numbers exactly as given."
```

Measure on InLegalBench. This is your zero-shot baseline.

Expected outcome:
- Case research quality: ~55–65% of target (good retrieval, generic reasoning)
- Legal terminology accuracy: ~70% (base models know legal terms)
- Argument quality: ~40–50% of target (generic, not advocate-style)

**Why this matters:** If zero-shot baseline is already at 60%+ and LoRA gets you to 70%, the 10% gain may not justify 6 weeks of data prep. If zero-shot is at 40% and LoRA gets you to 70%, LoRA is essential.

**Decision rule:**
- Zero-shot > 65% on case research → LoRA is optional enhancement, not critical path
- Zero-shot < 55% on case research → LoRA is necessary, proceed with training
- Argument generation zero-shot < 50% → always train argument-gen adapter regardless

---

### 8.7 Knowledge Cutoff and BNS/BNSS Gap

Qwen2.5-7B cutoff: late 2024. BNS/BNSS/BSA were passed December 2023, operationalised July 2024.

The base model may have partial BNS knowledge from web text, but it will be incomplete and potentially wrong on exact section numbers.

**Rule:** Never trust LoRA memory for statutory content. Always retrieve via PageIndex.

LoRA training on BNS Q&A pairs teaches the model how to *reason about* BNS, not to *recall* it. The actual section text always comes from PageIndex retrieval. This is the architectural guarantee against hallucination.

**Hallucination mitigation strategy:**
1. System prompt: "You must only cite sections that appear in the retrieved context below. Do not cite sections from memory."
2. Post-generation validation: After every response, run a section citation extractor. For each cited section (e.g., "BNS Section 103"), verify it exists in the PageIndex response. Flag or regenerate if citation is not grounded.
3. Critic model scores Evidence Grounding (weight 0.25) — directly penalises ungrounded claims.
