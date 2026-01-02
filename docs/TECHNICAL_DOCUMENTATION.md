# Technical Documentation – Blood Cancer AI System (v4)

## 1. Project Description
This project implements an **AI-assisted blood cancer detection and explanation system** for **blood smear microscopy images**.
It is designed as a **research / clinical decision-support prototype** and includes:

- Deep learning classification (binary or multi-class)
- Explainability (Grad-CAM)
- Agentic orchestration (Supervisor + tool-using agents) using LangGraph
- Grounded report generation (template + optional LLM drafting) with local RAG
- Human-in-the-loop labeling and continual improvement retraining
- Web API + UI dashboard

**Primary use-case (v4):**
- Screen blood smear images for **leukemia-like patterns** and prioritize uncertain cases for expert review.

---

## 2. Goals and Non-goals

### 2.1 Goals
- Provide a reproducible and modular architecture for detection + explanation
- Enable auditable agentic workflows (plan + trace)
- Provide practical MLOps patterns (case store, labeling, retraining jobs)
- Provide a UI for clinicians/researchers to explore results and manage feedback

### 2.2 Non-goals
- Not a diagnostic medical device
- Not trained/validated for clinical deployment without extensive governance and trials
- Not a substitute for pathology/hematology experts

---

## 3. System Architecture

### 3.1 Components
1) **UI Dashboard (Streamlit)** – upload, browse buckets, review, label, retrain, inspect trace  
2) **API Server (FastAPI)** – prediction endpoint + case store + retraining triggers  
3) **Case Store (SQLite)** – persists cases, predictions, traces, labels, stats  
4) **Agentic Core (LangGraph + Supervisor)** – orchestrates all steps  
5) **Model Layer (PyTorch + timm)** – classifier backbone and checkpoints  
6) **Explainability Layer (Grad-CAM)**  
7) **Grounding (TF-IDF RAG)** – local `knowledge_base/` retrieval  
8) **Training Layer** – baseline training + incremental retraining on feedback  

### 3.2 Data Flow
**Upload → Supervisor plan → tool execution → persist results → UI**

**Labeling → feedback dataset → retraining job → updated model checkpoint**

---

## 4. Agentic AI Design

### 4.1 Why “Agentic”
The system uses an explicit agentic pattern:
- A **Supervisor** decides *what to do next* (plan) and executes steps as tools.
- Each tool has a clear **purpose**, **IO contract**, and **guardrails**.
- Every tool invocation is recorded in an **audit trace**.

### 4.2 Supervisor
**File:** `src/bloodcancer/agents/supervisor.py`

Responsibilities:
- Determine routing/model policy (`routing` tool)
- Create ordered tool plan
- Execute plan with tracing
- Optional LLM planning (guardrailed; never medical interpretation)

### 4.3 Tools
**File:** `src/bloodcancer/agents/tools.py`

Tools include:
- routing
- triage
- preprocess_qc
- inference
- active_learning
- explain
- rag
- report
- safety_gate

Tools that wrap explicit agents:
- triage → `TriageAgent`
- preprocess_qc → `PreprocessQCAgent`
- inference → `InferenceAgent`
- explain → `ExplainabilityAgent`
- rag → `EvidenceRAGAgent`
- report → `ReportAgent`
- safety_gate → `SafetyGateAgent`

### 4.4 Auditability
The system stores:
- the tool plan
- per-tool start/done/error events
- key outputs produced
in `cases.trace_json` so a reviewer can understand *how* the outcome was produced.

---

## 5. Modeling

### 5.1 Baseline classifier
- Backbones from **timm** (e.g., EfficientNetV2)
- Loss: cross-entropy
- Metrics: accuracy, macro-F1, optional AUC for binary

### 5.2 Explainability
- Grad-CAM overlay image is saved under `artifacts/explanations/`
- Summary statistic is recorded in output

### 5.3 Uncertainty & Active Learning
Active learning uses:
- uncertainty = 1 - max_prob
- margin = top1 - top2
- score = uncertainty + 0.5*(1 - margin)

Cases are suggested for labeling if:
- confidence is below a high threshold OR
- margin is small (ambiguous)

---

## 6. Grounded Explanations (RAG + Templates)

### 6.1 Knowledge base
- `knowledge_base/` contains short markdown notes about:
  - dataset limitations
  - explainability limitations
  - clinical disclaimers

### 6.2 Retrieval
- TF-IDF retrieval (scikit-learn) returns top-k evidence snippets
- This ensures explanations remain grounded and consistent

### 6.3 Report generation
- Template in `configs/prompts/report_template.md`
- Optional LLM drafting augments *interpretation text only* under strict constraints:
  - non-diagnostic
  - uncertainty
  - clinician review required
  - no treatment advice

---

## 7. API Design

### 7.1 Key endpoints
- `POST /predict` – upload image, run pipeline, persist case
- `GET /cases` – list cases (filters: predicted_label, needs_review, suggested_for_labeling)
- `GET /cases/{id}` – get case + decoded probs + trace
- `POST /cases/{id}/label` – store human label and copy image to feedback set
- `GET /stats` – aggregate stats
- `POST /train/trigger` – start retraining job
- `GET /agents`, `GET /tools` – inspect agentic design

---

## 8. UI Dashboard

### Tabs
- Upload & Predict
- Buckets by predicted type
- Cases table
- Stats
- Review & Label
- Active Learning
- Retrain / continual learning
- Agents
- Tools

The UI is intentionally simple and auditable; it uses the API as source-of-truth.

---

## 9. Tests

### Current tests
- import smoke tests
- agent specs availability
- tool specs availability (via API or import-level validation)

### Suggested additional tests (roadmap)
- golden-case inference tests with fixed seed
- data preparation tests on a tiny fixture dataset
- regression tests for report templates
- API integration tests (testclient)

---

## 10. Security, Safety, and Governance

- Never store PHI in logs.
- Keep case storage local or encrypted at rest in production.
- Maintain audit trails (plan + trace + model version).
- Use confidence gating + human review.
- Validate on institution-specific data before any real-world use.

See: `docs/SECURITY_SAFETY.md`

---

## 11. Improvements / Roadmap
See: `docs/ROADMAP.md`

High-impact improvements:
- calibration (temperature scaling) + abstention
- segmentation / detection of WBCs and blast cells
- patient-level aggregation from multiple fields-of-view
- domain adaptation and stain normalization
- model registry + monitoring + drift alerts
- clinical-grade evaluation with external validation

---

## 12. References
This repository uses and is inspired by:
- PyTorch (training/inference)
- timm (vision backbones)
- pytorch-grad-cam (explainability)
- FastAPI (serving)
- Streamlit (dashboard)
- LangGraph (agentic orchestration)

For blood smear leukemia datasets, look for:
- C-NMC 2019 (ALL classification challenge datasets)
- ALL-IDB (classic leukemia image DB)

---

## 13. Usage Summary
1) Download/prepare dataset  
2) Train model  
3) Start API  
4) Start UI  
5) Upload cases  
6) Label uncertain cases  
7) Retrain on feedback  

---

## 14. Conclusion
This repository provides a complete, modular foundation for an **agentic AI-assisted blood cancer detection and explainability system** with a practical human-in-the-loop loop. It is suitable for research and prototyping and can be extended into more advanced clinical decision-support workflows with appropriate validation.


---

## 15. Multi-Model Routing & Ensembles (v5)
The system supports separate checkpoints for:
- Stage-1 screening (normal vs leukemia)
- Stage-2 subtyping (ALL/AML/CLL/CML), conditional on stage-1 output
- Ensembles (averaging probabilities across multiple checkpoints)

This is controlled by:
- `configs/models.yaml` (policies + candidates + ensembles)
- `scripts/train_policies.py` (multi-model trainer)
- `artifacts/model_registry.json` (auto-registered checkpoints and metrics)

See: `docs/MULTI_MODEL_PIPELINES.md`
