# Indian Legal AI Agent

Domain-aware legal research assistant for Indian law. Uses QLoRA-trained adapters on Qwen2.5-7B-Instruct across 12 legal domains, with Claude Sonnet for argument generation and Claude Haiku for routing.

## Architecture

```
Document (PDF / Image / Text)
        ↓
   OCR (GLM-OCR) + PageIndex Builder
        ↓
   Summarizer — Legal-BERT + PEGASUS (Claude Haiku fallback)
        ↓
   Domain Router — Claude Haiku → confidence scores
        ↓
   Adapter Selector — picks top domain adapters (threshold 0.40)
        ↓
   LoRA Engine — Qwen2.5-7B-Instruct + QLoRA adapter (4-bit NF4)
        ↓
   Citation Validator — tags unverified BNS/IPC references
        ↓
   Research Answer
        ↓ (if query_type="arguments")
   Argument Generator — Claude Sonnet (CoT + Vision)
        ↓
   Prosecution / Defense / Both — structured JSON with citations
```

## 12 Legal Domains

| Domain | Coverage |
|--------|----------|
| `criminal_violent` | Murder, culpable homicide, BNS 100–115 |
| `criminal_property` | Theft, robbery, dacoity, BNS 300–330 |
| `kidnapping_trafficking` | Abduction, trafficking, BNS 137–144 |
| `sexual_offences` | POCSO, rape, BNS 63–79 |
| `land_property` | Acquisition, encroachment, title disputes |
| `family_matrimonial` | Divorce, custody, maintenance, DV Act |
| `constitutional` | Fundamental rights, writs, Articles |
| `corporate_commercial` | Companies Act, contracts, insolvency |
| `labour_employment` | Factories Act, POSH, wrongful termination |
| `cyber_digital` | IT Act, cybercrime, data privacy |
| `tax_fiscal` | GST, income tax, TDS disputes |
| `civil_general` | CPC, limitation, torts |

## Models

| Task | Model |
|------|-------|
| OCR | `zai-org/GLM-OCR` |
| Summarization (primary) | Legal-BERT + PEGASUS |
| Summarization (fallback) | Claude Haiku |
| Domain routing | Claude Haiku |
| Legal QA | Qwen2.5-7B-Instruct + QLoRA adapter |
| Argument generation | Claude Sonnet (CoT + Vision) |

## Stack

- **API** — FastAPI + Uvicorn
- **UI** — Gradio 6
- **Databases** — MongoDB (case indexes), PostgreSQL (audit logs), Redis (cache + sessions)
- **Storage** — MinIO (documents)
- **Training** — QLoRA via PEFT + TRL on Google Colab / RunPod

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env   # then fill in your keys
```

Required values in `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
POSTGRES_PASSWORD=changeme
MINIO_SECRET_KEY=changeme
API_KEY=your-chosen-api-key
```

### 2. Run (CPU / Mac)

```bash
docker compose -f docker/docker-compose.yml up --build
```

### 3. Run (Linux + NVIDIA GPU)

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up --build
```

### 4. Rebuild a single service (fast)

```bash
docker compose -f docker/docker-compose.yml up --build gradio
docker compose -f docker/docker-compose.yml up --build api
```

### Services

| Service | URL |
|---------|-----|
| Gradio UI | http://localhost:7860 |
| REST API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MinIO console | http://localhost:9001 |

## API Usage

All endpoints require `x-api-key` header matching `API_KEY` in `.env`.

### Create a case
```bash
curl -X POST http://localhost:8000/cases \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"title": "State vs. Sharma — BNS 103"}'
```

### Upload a document
```bash
curl -X POST http://localhost:8000/cases/{case_id}/documents \
  -H "x-api-key: your-api-key" \
  -F "file=@fir.pdf"
```

### Research query
```bash
curl -X POST http://localhost:8000/cases/{case_id}/query \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "query": "What BNS sections apply for murder with common intention?",
    "query_type": "research"
  }'
```

### Generate arguments
```bash
curl -X POST http://localhost:8000/cases/{case_id}/query \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "query": "Accused found with stolen vehicle. FIR under BNS 303.",
    "query_type": "arguments",
    "side": "defense"
  }'
```

## Training Adapters

### Train all 12 domains (RunPod / Colab)

```python
from training.colab_train import train_all_adapters

train_all_adapters(
    epochs=2,
    batch_size=4,
    max_seq_length=1024,
    early_stopping_patience=1,   # stops immediately if loss rises
)
```

### Train a single domain

```python
from training.colab_train import train_on_colab

train_on_colab(domain="criminal_violent", epochs=3, batch_size=4)
```

### Install dependencies (Colab / RunPod)

```bash
pip install "torch>=2.4.0" "transformers==4.46.3" "peft==0.13.2" \
            "trl==0.12.2" "accelerate==0.34.2" "bitsandbytes==0.43.3" \
            "datasets>=2.18.0" "sentencepiece>=0.2.0" "pyyaml>=6.0" "sympy==1.12"
```

## Project Structure

```
├── api/                  FastAPI app + routes
│   ├── main.py           Lifespan, middleware, adapter preload
│   └── routes/           cases, documents, query endpoints
├── core/
│   ├── ingestion/        OCR, summarizer, pageindex builder
│   ├── routing/          Domain router + adapter selector
│   ├── reasoning/        LoRA engine, case research, argument generator
│   ├── indexing/         PageIndex query
│   └── validation/       Citation validator
├── db/                   MongoDB, PostgreSQL, Redis clients
├── training/
│   ├── colab_train.py    train_on_colab() + train_all_adapters()
│   └── configs/          Per-domain YAML configs
├── ui/app.py             Gradio 6 interface
├── docker/
│   ├── docker-compose.yml
│   ├── docker-compose.gpu.yml   GPU override (NVIDIA only)
│   └── Dockerfile.api
└── adapters/             Trained LoRA adapters (gitignored)
```
