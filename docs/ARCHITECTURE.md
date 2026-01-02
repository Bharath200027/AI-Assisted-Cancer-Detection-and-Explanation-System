# Architecture (v1)

## Flow
1. **Upload blood smear image**
2. **Agentic pipeline** (LangGraph)
   - triage → preprocess QC → inference → Grad-CAM → RAG → report → safety gate
3. **Return**: prediction + confidence + explanation overlay + structured report

## Extend to multi-task
- Add **cell detection/segmentation** (e.g., WBC detection with morphological attributes datasets like MICCAI 2024 resources).
- Add MIL for slide/case-level aggregation if you have multiple fields-of-view per patient.

## Monitoring / QA (recommended)
- Log: model version, preprocessing, confidence, retrieved evidence, report text
- Drift detection: embeddings or image stats
- Human-in-the-loop feedback loop
