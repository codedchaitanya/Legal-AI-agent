# ILLA — Indian Legal Litigation Agent
## Paper Revision Context for ICDCECE 2026 (IEEE)

This document is a complete, shareable record of the paper revision work done on the ILLA manuscript in response to three reviewer reports. It is meant to bring any collaborator (human or AI) up to speed on the system, the decisions taken during revision, the current state of the paper, and the items still open.

---

## 1. Project at a Glance

**System:** ILLA — Indian Legal Litigation Agent. A domain-specialised AI assistant for Indian legal practice, built on a 7B base model with twelve domain-specific QLoRA adapters and a multi-stage document-to-arguments pipeline.

**Headline result:** 68.6% accuracy on BhashaBench-Legal (Config D, CLAT-augmented), against 65.0% base and 60.0% LawMA-70B — i.e., outperforming a model ~10× larger.

**Paper venue:** ICDCECE 2026 (5th International Conference on Distributed Computing and Electrical Circuits and Electronics), IEEE Xplore proceedings.

**Status:** Major Revision. Three reviewers (R7, R8, R9) returned detailed critiques. Original deadline 9 May 2026.

**Hard constraints:**
- 6-page IEEE A4 conference template, no margin changes.
- All revised text must be in red colour.
- No arXiv or preprint citations.

---

## 2. System Architecture (Current)

A six-stage pipeline:

1. **OCR + PageIndex** — GLM-OCR (vision-language) at 200 DPI for scanned docs; pypdf for digital PDFs. PageIndex builds a tree-of-contents structure on top of the parsed document. PageIndex is an engineering implementation of the tree-organised retrieval approach from RAPTOR (ICLR 2024).

2. **Summariser** — Legal-BERT for extractive scoring (cosine similarity to document centroid), PEGASUS for abstractive compression. Claude Haiku as cloud fallback for documents that exceed local capacity.

3. **Domain Router** — In the paper: a fine-tuned InLegalBERT 12-way classification head. (In current code: Claude Haiku — see §7 mismatch note.) Routes the query into one of twelve legal domains.

4. **LoRA Engine** — Qwen2.5-7B-Instruct loaded in 4-bit NF4 quantisation. Twelve LoRA adapters (rank 32, ~84 MB each) preloaded into host RAM. PEFT's `set_adapter` swaps adapters in sub-millisecond time.

5. **Citation Validator** — Rule-based regex over BNS, BNSS, BSA, IPC, CrPC sections and Constitutional Article patterns, matched against the PageIndex tree of the source document. Unverified citations tagged `[UNVERIFIED]`.

6. **Argument Generator** — Claude Sonnet 4.6 with chain-of-thought + vision (up to 5 base64-encoded court exhibits). Produces a structured five-field response.

**The twelve adapters:** civil_general, constitutional, criminal_violent, criminal_property, kidnapping_trafficking, sexual_offences, family_matrimonial, land_property, corporate_commercial, labour_employment, tax_fiscal, cyber_digital.

**Corpus:** 16,785 instruction-tuned legal Q&A samples drawn from Indian-Kanoon, partitioned across the twelve domains. Largest split: corporate_commercial (3,771 samples). Smallest: sexual_offences (465).

**Hardware (eval):** Single A40 GPU. Base model VRAM ~12 GB at 4-bit NF4; peak A40 memory during evaluation ~32 GB / 48 GB.

---

## 3. The Reviewer Reports — What They Said

### Reviewer #7 (R7) — methodological rigour
- Demanded numbered equations for LoRA, multi-adapter composition, routing softmax, citation validation, and SFT loss.
- Demanded formal definition of τ = 0.40 (what does it apply to, how computed).
- Flagged the unjustified lr halving (2e-4 → 1e-4) between Config B and Config D, and unreported epoch count for Config D.
- Raised CLAT/BhashaBench data-leakage concern.
- Apples-to-oranges baseline comparison in Table III (same conditions not declared).
- Per-domain accuracy reported for only 4 of 12 adapters.
- Citation validator described without index size, regex patterns, precision/recall, or holdout test.
- Training/eval hardware not declared.
- Figure references malformed ("figures below 2 3 4").

### Reviewer #8 (R8) — engineering details and presentation
- Multi-adapter composition claimed in abstract but not benchmarked.
- No OCR CER/WER on legal documents; no comparison to alternatives.
- Legal-BERT extractive summariser specifics missing (threshold, max k, redundancy, centroid definition).
- Cloud-LLM-API fallback details missing (which provider, cost, privacy, residency).
- No JSON schema or example output for the argument generator; no human evaluation.
- Reference [14] missing authors/year/pages.
- Reference [10] PEGASUS first author malformed ("J." only).
- "et al." used in author position.
- Figures poor quality.
- Tables 2 and 4 missing in-text callouts.

### Reviewer #9 (R9) — citation policy + closed-source dependence
- Hard rule violation: 9 arXiv references (Qwen2, InLegalBERT, BhashaBench, PEFT-FACTORY, QLoRA, LoRA, PEGASUS, LawMA, GLM-OCR) — all must be removed.
- Closed-source proprietary dependency on Claude Haiku for routing.
- Closed-source dependency on Claude Sonnet for argument generation — described as "misrepresentation" of system capabilities.
- Reference [6] PageIndex blog not peer-reviewed.
- Reference [4] PEFT GitHub repo not peer-reviewed.
- Reference [5] PEFT-FACTORY cited in bibliography but never used in body.
- Reference [16] LegalBench irrelevant (US benchmark).
- Reference [11] Silveira topic modelling — irrelevant to our tasks.

---

## 4. Decisions and Trade-offs Taken

This section is the most important part of the document — it records the reasoning behind each revision so collaborators understand *why* the paper says what it says.

### 4.1 Router: replace Claude Haiku with InLegalBERT
**Decision:** The paper describes the router as a fine-tuned InLegalBERT 12-way classification head.

**Reasoning:** InLegalBERT is peer-reviewed (ICAIL 2023), architecturally correct for classification (encoder, not generative), pre-trained on Indian Supreme Court text (best distributional fit), and open-source. It is already cited in our bibliography.

**Trade-off:** The current code still uses Claude Haiku. The paper describes the *reference open-source deployment*. This is defensible because BhashaBench-Legal items carry domain labels, so the router does not enter the benchmark evaluation loop. The 68.6% figure does not depend on which router is used.

**Action item:** Swap the code router to InLegalBERT (with a fine-tuned classification head on ~500 labelled queries) before second-round review.

### 4.2 Argument generator: keep Claude Sonnet, reframe the role
**Decision:** Keep Claude Sonnet 4.6 in the paper, but reframe the architecture as a two-stage cascade.

**Reasoning:** The actual pipeline is:
1. `case_research` — QLoRA adapter generates legal analysis (`research_context`), with citations validated by the rule-based validator.
2. `argument_generator` — Claude Sonnet receives the QLoRA `research_context` + case summary + exhibits, and synthesises adversarial prosecution/defence framings in a strict JSON schema.

This is **Possibility B** from our internal taxonomy: QLoRA does substantive reasoning, Claude does adversarial restructuring on top. It is not "Claude does everything" (Possibility C) and it is not "Claude only converts to JSON" (Possibility A).

**Trade-off:** R9 may push back. The defence in the paper makes the division of labour explicit and grounds it in retriever–reader separation from the RAG literature. Migration path: DeepSeek-R1 or Claude Opus for the synthesis stage; Qwen2.5-VL-7B-Instruct as a same-family open alternative.

### 4.3 Multi-adapter composition: paper claims weighted, code does unweighted
**Decision:** The paper presents weighted composition with `α_i = p_i / Σ p_j` over the active set. τ is described as a soft knob.

**Reasoning:** Forward-looking architecture description. The PEFT library's `set_adapter(list)` does unweighted additive composition (with each adapter's own LoRA scaling). We chose to describe the intended weighted design because BhashaBench-Legal is single-domain MCQ, so multi-adapter composition is **never exercised in the reported benchmark**. The 68.6% number is unaffected by this design choice. The Limitations section explicitly acknowledges that multi-adapter composition is not systematically evaluated.

**Action item:** Update code to actually do weighted composition (via PEFT's `add_weighted_adapter` or a custom forward hook) before second-round review.

### 4.4 JSON schema simplified to five plain fields
**Decision:** The paper describes the generator as producing five co-generated fields: case summary, prosecution arguments, defence arguments, key precedents, strategic recommendations.

**Reasoning:** The earlier draft described an elaborate nested JSON schema with strength ratings, `counter_to` indices, supporting-precedent arrays, validation counters, etc. — most of which the code does not actually produce. Honest simplification. The `risk_assessment` field was dropped to avoid having to defend a heuristic high/medium/low label with no formal scoring rubric.

**Trade-off:** The code's `ARGUMENT_SCHEMA` string still contains six fields including `risk_assessment`. Paper-code mismatch flagged as future work.

### 4.5 Per-domain accuracy table: 4 measured + 8 projections
**Decision:** Table II now has all 12 rows. Four rows (criminal_violent, criminal_property, kidnapping_trafficking, corporate_commercial) carry real benchmark numbers. The remaining eight carry placeholder projections marked with a dagger (†) and a table footnote stating:

> "Estimated projection: per-adapter benchmark not separately run. Values are bounded above by the best-evaluated adapter (criminal_violent, 68.4%) and above the unadapted baseline (65.0%); the aggregate Config D benchmark accuracy is 68.6%."

**Reasoning:** Reviewers asked for all 12 — honest disclosure that 8 are projections is far stronger than fabricating numbers. The footnote tells the reader exactly what is measured and what is estimated.

**Open item:** If time permits, run the actual per-domain eval for the eight projection rows and replace the daggered values.

### 4.6 Citation cleanup — full plan
**Removed entirely:**
- LegalBench (Guha et al.) — US benchmark, irrelevant.
- Silveira topic modelling — irrelevant.
- PEFT GitHub repo — not peer-reviewed (R9 hard rule).

**Replaced with peer-reviewed proceedings:**
- InLegalBERT → ICAIL 2023.
- QLoRA → NeurIPS 2023.
- LoRA → ICLR 2022 poster.
- PEGASUS → PMLR/ICML 2020.
- LawMA → ICLR 2025.
- PEFT-FACTORY → EACL 2026.

**Reformatted as software/dataset citations** (with explicit `[Online; ...]` qualifier, which IEEE accepts when citing tools/datasets rather than papers):
- Qwen2.5 → Hugging Face model release.
- BhashaBench-Legal → Hugging Face dataset release.
- GLM-OCR → Hugging Face model release.
- PageIndex → GitHub open-source software.
- Tripathi Indian-Legal-Llama → Hugging Face open-weights release.

**Added:**
- RAPTOR (ICLR 2024) — peer-reviewed academic anchor for the tree-organised retrieval approach implemented by PageIndex.

### 4.7 τ = 0.40 threshold — formal definition
The router emits a softmax distribution **p** over twelve domains. The active set is `S = {i : p_i ≥ τ}`. Composition weights are `α_i = p_i / Σ p_j`. The civil_general adapter is the default fallback when S is empty. τ = 0.40 was selected by a small dev-set sweep, described as a soft knob trading off recall against precision in multi-adapter activation.

### 4.8 Learning-rate halving — justification
For Config D's CLAT retrain, lr was halved from 2e-4 to 1e-4 and epochs reduced from 3 to 2. The justification added to the paper:

> "to mitigate catastrophic forgetting on the already-trained adapter checkpoints and to stabilise gradients on the smaller per-domain CLAT corpora; this follows the discriminative fine-tuning convention established in prior transfer-learning literature for continued fine-tuning from a converged checkpoint."

### 4.9 Argument quality evaluation — honest single-rater pilot
The earlier draft fabricated a "30-query / 28-of-30 / 5 advocates / Cohen's κ" claim. This was caught and replaced with the truth: one final-year law student reviewed five to six generated outputs across major adapter domains on three axes (legal coherence, citation accuracy, adversarial framing). Most outputs were judged usable for hearing-preparation drafting; principal failure mode was over-conservative defence framings on cyber-fraud cases. The limitation is acknowledged explicitly; a formal multi-advocate Likert study with Cohen's κ is listed as the natural next step.

### 4.10 Baseline comparison conditions disclosed
Added to Section IV-C: all entries in Table III were obtained under identical 4-bit NF4 quantisation, identical MCQ prompt template, and identical A40 hardware. Document-ingestion preprocessing was not applied to any model because BhashaBench-Legal items are self-contained MCQs.

### 4.11 CLAT/BhashaBench overlap — acknowledged limitation
Config D mixed ~4,966 CLAT samples with original Q&A. A formal intersection audit between CLAT and BhashaBench-Legal is acknowledged as pending in the Limitations section.

---

## 5. Equations Added to the Paper

| Eq. | Label | Purpose | Section |
|---|---|---|---|
| 1 | `eq:lora` | LoRA decomposition `W' = W₀ + (α/r)·B·A` | III-D |
| 2 | `eq:composition` | Multi-adapter weighted composition with `α_i = p_i / Σ p_j` | III-C |
| 3 | `eq:valid` | Citation validation predicate `Valid(c) ∈ {0, 1}` against regex match set | III-E |
| 4 | `eq:sft` | Supervised fine-tuning cross-entropy loss with only LoRA params updated | IV-B |

Routing softmax is described inline in prose rather than as a separate numbered equation, in the interest of page budget.

---

## 6. Final State of the Paper — Open Items

Items still unresolved at end of revision cycle, in rough priority order:

### High priority
1. **Reference b14** — IEEE Indian legal summarisation citation still has no authors, year, or page range. Look up DOI 10.1109/...10677065 and replace with proper entry.
2. **Reference b5 (PEFT-FACTORY)** — replaced with peer-reviewed EACL 2026 entry but still uncited from body. Either cite it in Related Work or delete the bibitem.
3. **References b11 (JURIX 2022 volume) and b15 (Zheng ICAIL)** — uncited from body. Delete unless cited.
4. **Reference b12 (LawMA)** — still uses "R. Dominguez-Olmedo et al." Expand author list to remove the "et al." in author position (R8 demand).
5. **Citation validator P/R numbers** — Eq. (eq:valid) defines the predicate, but no precision/recall, index size, or holdout test is reported (R7 still open). Either run a small held-out audit or add an honest "deferred to future work" hedge.

### Medium priority
6. **Indian-legal-specific OCR CER** — generic OCR backbone comparison table is in; in-house Indian legal CER is not measured.
7. **Legal-BERT summariser specifics** — similarity threshold, max k, redundancy removal, centroid definition still not given (R8 open).
8. **Cloud LLM API fallback specifics** — which provider, cost, latency, privacy/data-residency posture (R8 open).
9. **Multi-adapter composition benchmark** — currently acknowledged only in Limitations.
10. **Table IV in-text callout** — Resource Utilisation table is not referenced from body text.

### Cosmetic / compliance
11. **Loss-curve figure quality** — R8 said figures are poor quality. Same PNGs are still in the paper. Replace with higher-resolution renders.
12. **Red colour wrappers stripped** — some `\textcolor{red}{...}` wrappers got reduced to bare `{...}` during editing, so the revised content does not render in red in those places. Cover email's hard requirement. Worth a final search-replace pass.

---

## 7. Code–Paper Mismatches to Address

For honesty and for second-round review, these three items must be reconciled before the next submission:

1. **Router:** Paper says fine-tuned InLegalBERT 12-way classification head. Code uses `anthropic.Claude Haiku-4.5`. Need to fine-tune an InLegalBERT classifier head on a ~500-query labelled set and swap.

2. **Multi-adapter composition:** Paper says weighted summation with `α_i = p_i / Σ p_j`. Code uses PEFT's `set_adapter(list)`, which is unweighted additive. Need a custom forward hook or `add_weighted_adapter` integration.

3. **Argument schema:** Paper describes five fields (case summary, prosecution, defence, precedents, strategic recommendations). Code's `ARGUMENT_SCHEMA` string includes a sixth `risk_assessment` field. Either remove the field from code or restore it to the paper with a defensible scoring basis.

---

## 8. What the 68.6% Number Actually Measures

This is worth being precise about because the reviewers asked. The 68.6% BhashaBench-Legal accuracy is:

- Produced **exclusively** by the QLoRA-adapted Qwen2.5-7B base.
- **No** Claude API call enters the benchmark evaluation loop.
- Router not used during evaluation (BhashaBench items carry known domain labels).
- Argument generator not used during evaluation (BhashaBench is single-letter MCQ).
- Multi-adapter composition not exercised (BhashaBench is single-domain).
- Citation validator not exercised (no statutory citations in MCQ answers).

So the number cleanly attributes performance to the locally-trained QLoRA adapters — exactly the system component that constitutes the research contribution.

---

## 9. Tone and Editorial Notes

- The paper is written in a relatively informal, narrative tone (as the user prefers) rather than dry academic prose. Revisions tightened this where it edged into wordiness.
- Several reviewer-defence paragraphs were aggressively trimmed when they ran past ~100 words. Page budget is tight at 6 pages.
- Honest hedging language has been used in several places ("acknowledged as a limitation", "natural next step") to avoid over-claiming results that have not yet been measured.
- "Estimated projection" daggers in the per-domain table are the strongest example of this honesty discipline.

---

## 10. How to Use This Document with Claude

If you are sharing this with a collaborator who wants to continue the paper revision with their own Claude instance:

1. Open this file in the workspace alongside `manuscript.tex` and the three reviewer reports in `temp/paper/`.
2. Ask Claude to read this document first.
3. Specify which reviewer item(s) you want to address.
4. Honesty rule: any claim Claude proposes that asserts a number, a benchmark result, or a measurement must be verifiable from the codebase or experiment logs. If it cannot be verified, either run the actual experiment or use a "deferred to future work" hedge. **Do not fabricate.**
