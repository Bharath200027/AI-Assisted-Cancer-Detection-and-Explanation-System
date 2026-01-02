# Multi-Model Pipelines, Two-Stage Routing, and Ensembles (v5)

## 1) What “true separate pipelines” means here
This repository supports *separate* model checkpoints and processing logic for:
- **Stage 1: Screening** (Normal vs Leukemia)
- **Stage 2: Subtyping** (ALL / AML / CLL / CML) — executed only when stage 1 predicts leukemia
- Optional **ensemble inference** for either stage

This is configured in `configs/models.yaml` and executed by the **Supervisor** via tools:
- `routing` → selects candidates
- `inference` → stage-1 inference (single or ensemble)
- `subtype_inference` → stage-2 inference if applicable (single or ensemble)

## 2) Configuration
Edit `configs/models.yaml`:

- `default_policy: two_stage`
- `policies.two_stage.stage1.candidates`: screening models
- `policies.two_stage.stage2.candidates`: subtype models
- Set `ensemble: true` for averaging across multiple checkpoints.

Each candidate needs:
- `id`
- `model_name` (timm model)
- `checkpoint` path
- `class_names`

## 3) How the system routes
- The **RoutingTool** selects a policy and stage candidates.
- Stage-1 inference runs first.
- If stage-1 predicted label is `"leukemia"`, the supervisor runs **stage-2 subtype inference**.
- Outputs:
  - `predicted_label` + `confidence` + `probs` (stage-1)
  - `predicted_subtype` + `subtype_confidence` + `subtype_probs` (stage-2)
  - `disagreement` / `subtype_disagreement` if ensemble enabled

Stage-1 results are preserved in `stage1_snapshot` for audit/reporting.

## 4) Training multiple policies
Use `configs/training.yaml` to map policies/stages to dataset folders, then run:
```bash
python scripts/train_policies.py --epochs 10 --batch_size 32 --img_size 224
```

This:
- trains each configured candidate
- copies the best checkpoint to the configured `checkpoint` path
- registers it in `artifacts/model_registry.json`

## 5) Automatic checkpoint registration & selection
- Training registers checkpoints into: `artifacts/model_registry.json`
- Routing prefers the **best** registry entry (by best_accuracy) if checkpoint exists.

You can also manually register a checkpoint:
```bash
python scripts/register_checkpoint.py   --id subtype_vit --policy two_stage --stage stage2   --model_name vit_base_patch16_224   --checkpoint artifacts/models/subtype/vit_best.pt   --class_names all aml cll cml   --best_accuracy 0.88
```

## 6) Notes on subtype datasets
Subtype prediction requires subtype-labeled datasets (ALL/AML/CLL/CML).
If you only have binary labeled data, you can still use:
- `binary` policy, or
- `two_stage` with stage2 disabled/untrained (it will no-op).
