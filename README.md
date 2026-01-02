# Blood Cancer AI – Agentic Detection, Explainability, UI Dashboard & Continual Learning (v6)

> **Research / decision-support only. Not a medical device.**
> Outputs can be wrong. Always require clinician review. Never use this system as a standalone diagnosis.

This repository provides an **end-to-end**, **agentic** AI system for **blood cancer screening support** from **blood smear microscopy images** (e.g., leukemia). It includes:

- Deep learning **image classification** (binary or multi-class)
- **Explainability** (Grad-CAM overlays)
- **Agentic AI** orchestration using a **Supervisor + tool-using agents** (LangGraph)
- **RAG grounding** for safe, evidence-backed explanations (local knowledge base)
- **FastAPI** backend (upload/predict, case storage, labeling, retraining)
- **Streamlit UI** dashboard to view:
  - statistics, confidence histograms
  - buckets by predicted cancer type
  - active-learning labeling queue
  - execution trace (plan + tool trace)
  - retraining jobs


## New in v6

- **Hierarchical ensembles**: Stage-1 screening → Stage-2 family-specific subtype ensembles (lymphoid vs myeloid) with optional fallback.
- **Best-model-per-class** selection using **per-class F1** derived from the validation confusion matrix stored in the registry.
- **Expert EDA utilities**: `scripts/eda_dataset.py` and `scripts/make_family_datasets.py` to analyze data quality and build family datasets.
- **UI fixes**: complete dashboard tabs, Active Learning queue, subtype-aware buckets.

---

## Table of Contents
- [1. Architecture](#1-architecture)
- [2. Agentic AI Design](#2-agentic-ai-design)
- [3. Tech Stack](#3-tech-stack)
- [4. Setup](#4-setup)
- [5. Dataset Options](#5-dataset-options)
- [6. Training](#6-training)
- [7. Run Inference](#7-run-inference)
- [8. Run API](#8-run-api)
- [9. Run UI Dashboard](#9-run-ui-dashboard)
- [10. Human-in-the-loop Labeling](#10-human-in-the-loop-labeling)
- [11. Continual Learning & Retraining](#11-continual-learning--retraining)
- [12. Multi-class Blood Cancer Types](#12-multi-class-blood-cancer-types)
- [13. Tests](#13-tests)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Multi-Model Pipelines & Ensembles](#15-multi-model-pipelines--ensembles)
- [16. Documentation](#16-documentation)
- [16. License](#16-license)

---

## 1. Architecture

**High-level flow**
1) Upload image  
2) **Supervisor** proposes a plan and executes tools:
   - routing → triage → QC → inference → active_learning → explain → RAG → report → safety gate
3) Persist results to SQLite
4) UI shows buckets, stats, trace, and review queues
5) Human labels difficult cases → stored as feedback
6) Retrain on base dataset + feedback (manual or auto retrain policy)

Key components:
- `services/api/` FastAPI backend + SQLite case store
- `src/bloodcancer/` core ML + agentic workflow
- `ui/` Streamlit dashboard
- `knowledge_base/` grounding docs for RAG
- `scripts/` dataset download + preparation

---

## 2. Agentic AI Design

### 2.1 Supervisor (Autonomous Orchestrator)
**File:** `src/bloodcancer/agents/supervisor.py`

The **SupervisorAgent**:
- selects model/pipeline policy (routing)
- creates an **action plan** (list of tools)
- executes tools in order
- records a **tool trace** (audit log)
- optionally uses an **LLM planner** (guardrailed) to propose the plan (never diagnosing)

Environment variables:
- `SUPERVISOR_USE_LLM=1` (optional; uses LLM only for planning)
- `DISABLE_EXPLAIN=1` to skip explainability step

### 2.2 Tools (Executable Units)
**File:** `src/bloodcancer/agents/tools.py`

Tools wrap explicit agents + extra utilities:
- `routing` – select model checkpoint/policy (see `configs/models.yaml`)
- `triage` – validate request, set modality
- `preprocess_qc` – quality checks
- `inference` – model prediction
- `active_learning` – compute uncertainty score + suggest labeling
- `explain` – Grad-CAM overlay
- `rag` – retrieve grounded evidence from `knowledge_base/`
- `report` – generate structured report (optional LLM drafting)
- `safety_gate` – enforce disclaimers + confidence gating

Inspect via:
- API: `GET /agents`, `GET /tools`
- UI tabs: **Agents**, **Tools**

---

## 3. Tech Stack
- **Python 3.10+**
- **PyTorch** for training/inference
- **timm** for pretrained vision backbones (EfficientNetV2/ViT variants)
- **grad-cam** for explanations
- **LangGraph** for agentic orchestration
- **FastAPI** for backend API
- **SQLite** for case storage + stats
- **Streamlit** for dashboard UI
- **scikit-learn** for TF-IDF retrieval (RAG) and metrics

---

## 4. Setup

### 4.1 Create environment & install
```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 4.2 (Optional) Configure LLM for report drafting
Reports are template-based by default. To enable LLM drafting:
```bash
export USE_LLM=1
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

> The LLM is **never allowed** to diagnose; it only drafts cautious text grounded in model outputs and retrieved evidence.

---

## 5. Dataset Options

### Option A (Recommended): Hugging Face mirror of C-NMC 2019
```bash
python scripts/download_cnmc_hf.py --out data/raw/cnmc_hf
python scripts/prepare_dataset.py --in_dir data/raw/cnmc_hf/export --out_dir data/processed/cnmc
```

### Option B: Kaggle dataset (requires Kaggle token)
```bash
python scripts/download_kaggle.py --dataset <owner/dataset-slug> --out data/raw/cnmc_kaggle
python scripts/prepare_dataset.py --in_dir data/raw/cnmc_kaggle --out_dir data/processed/cnmc
```

Expected final structure:
```
data/processed/cnmc/
  train/normal/...
  train/leukemia/...
  val/normal/...
  val/leukemia/...
  test/normal/...
  test/leukemia/...
```

---

## 6. Training
```bash
python -m bloodcancer.train   --data_dir data/processed/cnmc   --model_name tf_efficientnetv2_s   --epochs 10   --batch_size 32   --img_size 224   --out_dir artifacts
```

Artifacts:
- `artifacts/models/best.pt`
- `artifacts/models/last.pt`
- `artifacts/history.json`

---

## 7. Run Inference (CLI)
```bash
python -m bloodcancer.infer   --checkpoint artifacts/models/best.pt   --image_path path/to/sample.png   --out_dir artifacts
```

---

## 8. Run API
```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /predict`
- `GET /cases`, `GET /cases/{id}`
- `POST /cases/{id}/label`
- `GET /stats`
- `POST /train/trigger`
- `GET /agents`, `GET /tools`

---

## 9. Run UI Dashboard
```bash
streamlit run ui/streamlit_app.py
```

UI features:
- Upload & predict (report + Grad-CAM + execution trace)
- Buckets by predicted type
- Stats dashboard
- Review queue
- **Active learning queue** (suggested labeling)
- Retrain trigger + job monitoring
- Agent/Tool inspectors

---

## 10. Human-in-the-loop Labeling
From UI **Review & Label** or **Active Learning** tab:
- choose a case → assign `human_label` → submit  
This copies the image into:
`data/feedback/<label>/...`

---

## 11. Continual Learning & Retraining

### Manual retraining (recommended)
Trigger from UI or:
```bash
curl -X POST http://localhost:8000/train/trigger \
  -H "Content-Type: application/json" \
  -d '{"epochs":3,"model_name":"tf_efficientnetv2_s"}'
```

### Auto retrain (optional)
Edit `configs/continual.yaml`:
- `auto_retrain: true`
- `auto_retrain_min_new_labels: 50`

This will auto-trigger retraining when enough labeled feedback examples exist.

---

## 12. Multi-class Blood Cancer Types
See: `docs/BLOOD_CANCER_TYPES.md`

To use multi-class:
1) Prepare dataset folders with classes, e.g. `normal, all, aml, cll, cml`
2) Update `configs/app.yaml` `class_names`
3) Train
4) UI will auto-display buckets/stats per class

---

## 13. Tests
```bash
pytest -q
```
Tests include imports + agent/tool specs.

---

## 14. Troubleshooting
- **Checkpoint not found**: train first or set `MODEL_CHECKPOINT`
- **Torch/torchvision mismatch**: this repo avoids torchvision to reduce version conflicts.
- **Grad-CAM missing**: ensure `grad-cam` installed
- **HF download fails**: ensure `datasets` installed and internet access

---

## 15. Documentation
- `docs/TECHNICAL_DOCUMENTATION.md` – full technical doc
- `docs/ARCHITECTURE.md` – system architecture overview
- `docs/BLOOD_CANCER_TYPES.md` – multi-class setup
- `docs/SECURITY_SAFETY.md` – safety & governance guidance
- `docs/ROADMAP.md` – future improvements

---

## 16. License
MIT for code. Dataset licenses vary; follow the dataset terms from the source platform.


---

## 15. Multi-Model Pipelines & Ensembles

This repo supports **true separate pipelines**:
- Stage 1: **screening** (normal vs leukemia)
- Stage 2: **subtyping** (ALL/AML/CLL/CML) only when screening predicts leukemia
- Optional **ensembles** for either stage

Configuration:
- `configs/models.yaml` (policies, checkpoints, ensembles)
- `configs/training.yaml` (policy/stage → dataset mapping)

Train all configured candidates and auto-register checkpoints:
```bash
python scripts/train_policies.py --epochs 10 --batch_size 32 --img_size 224
```

See:
- `docs/MULTI_MODEL_PIPELINES.md`

---

## 16. Documentation
- `docs/TECHNICAL_DOCUMENTATION.md`
- `docs/ARCHITECTURE.md`
- `docs/MULTI_MODEL_PIPELINES.md`
- `docs/BLOOD_CANCER_TYPES.md`
- `docs/SECURITY_SAFETY.md`
- `docs/ROADMAP.md`


### Policy training via API (optional)
If you run the API service, you can trigger multi-policy training from the UI or by calling:
- `POST /train/policies_trigger`

This runs `scripts/train_policies.py` in a background thread and writes:
- checkpoints to `artifacts/models/...`
- registry to `artifacts/model_registry.json`


### Grad-CAM dependency
This project can optionally use the PyPI package **`grad-cam`** (import name `pytorch_grad_cam`). Install with:
```bash
pip install grad-cam
```


> Note: run `pip install -e .` from repo root so Python can import `bloodcancer`.
