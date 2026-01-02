# Datasets & Data Linking (Research)

This repository expects datasets in a **torchvision ImageFolder** layout:

```
data/processed/<dataset_name>/
  train/
    <class_0>/*.png
    <class_1>/*.png
  val/
    <class_0>/*.png
    <class_1>/*.png
  test/
    <class_0>/*.png
    <class_1>/*.png
```

## Recommended datasets
- **C-NMC (ISBI 2019)** / similar microscopy image datasets for stage-1 screening (normal vs leukemia)
- Subtype datasets for **ALL / AML / CLL / CML** (if available) for stage-2.

> Licensing varies by dataset. Always verify the dataset license and attribution requirements.

## Linking datasets to training
Edit:
- `configs/training.yaml` — maps each policy stage to a dataset directory
- `configs/models.yaml` — specifies candidate model IDs + default checkpoint paths

Then run:
```bash
python scripts/train_policies.py --epochs 10
```

The script will:
1. Train each candidate model (per policy stage)
2. Copy best checkpoint into the configured `checkpoint:` path
3. Register metrics into `artifacts/model_registry.json`

## Hierarchical family datasets (optional)
If you have a 4-class subtype dataset (`ALL/AML/CLL/CML`) and want family-specific models:

```bash
python scripts/make_family_datasets.py --src data/processed/subtype --dst data/processed
```

This creates:
- `data/processed/subtype_lymphoid` with classes `ALL/CLL`
- `data/processed/subtype_myeloid` with classes `AML/CML`

Update `configs/training.yaml` accordingly.

## Exploratory Data Analysis (EDA)
Generate an EDA report with distribution + quality diagnostics:

```bash
python scripts/eda_dataset.py --data_dir data/processed/cnmc --out_dir artifacts/eda/cnmc
python scripts/eda_dataset.py --data_dir data/processed/subtype --out_dir artifacts/eda/subtype
```

Outputs:
- `report.md`
- class count charts
- montages for train/val/test splits
