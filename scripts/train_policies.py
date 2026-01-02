from __future__ import annotations
import argparse
import subprocess
import sys
import json
from pathlib import Path

import yaml

from bloodcancer.registry import register_checkpoint, ModelEntry, utc_iso

def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command: " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        return int(p.returncode)

def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def _iter_stage_jobs(policy_name: str, policy_cfg: dict, train_cfg: dict):
    """Yield (registry_stage, stage_cfg, data_dir, job_label)."""
    # stage1
    if "stage1" in policy_cfg and train_cfg.get("stage1", {}).get("data_dir"):
        yield ("stage1", policy_cfg["stage1"], Path(train_cfg["stage1"]["data_dir"]), "stage1")

    # stage2 (flat or hierarchical)
    if "stage2" not in policy_cfg:
        return
    st2 = policy_cfg["stage2"]
    mode = (st2.get("mode") or "flat").lower()

    if mode == "flat":
        registry_stage = st2.get("registry_stage") or "stage2"
        data_dir = Path(train_cfg.get("stage2", {}).get("data_dir", ""))
        if str(data_dir):
            yield (registry_stage, st2, data_dir, "stage2")
        return

    # hierarchical
    st2_train = train_cfg.get("stage2", {}) or {}
    # families
    families = st2.get("families") or {}
    for fam_name, fam_cfg in families.items():
        fam_train = st2_train.get(fam_name, {}) or {}
        data = fam_train.get("data_dir")
        if not data:
            continue
        registry_stage = fam_cfg.get("registry_stage") or f"stage2_{fam_name}"
        yield (registry_stage, fam_cfg, Path(data), f"stage2/{fam_name}")

    # fallback
    fb_cfg = st2.get("fallback") or {}
    fb_train = st2_train.get("fallback", {}) or {}
    fb_data = fb_train.get("data_dir")
    if fb_cfg and fb_data:
        registry_stage = fb_cfg.get("registry_stage") or "stage2"
        yield (registry_stage, fb_cfg, Path(fb_data), "stage2/fallback")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_cfg", type=str, default="configs/training.yaml")
    ap.add_argument("--models_cfg", type=str, default="configs/models.yaml")
    ap.add_argument("--out_root", type=str, default="artifacts/policy_runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    training = _load_yaml(args.training_cfg)
    models = _load_yaml(args.models_cfg)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_map = (training.get("policies") or {})
    policies = (models.get("policies") or {})

    for policy_name, policy_cfg in policies.items():
        if policy_name not in train_map:
            print(f"[SKIP] No training config for policy: {policy_name}")
            continue

        policy_train = train_map[policy_name] or {}
        for registry_stage, stage_cfg, data_dir, label in _iter_stage_jobs(policy_name, policy_cfg, policy_train):
            candidates = stage_cfg.get("candidates") or []
            if not candidates:
                print(f"[SKIP] {policy_name}/{label}: no candidates")
                continue
            if not data_dir.exists():
                print(f"[SKIP] {policy_name}/{label}: data_dir not found: {data_dir}")
                continue

            for cand in candidates:
                cand_id = cand["id"]
                model_name = cand.get("model_name", "tf_efficientnetv2_s")
                ckpt_target = Path(cand.get("checkpoint", f"artifacts/models/{policy_name}/{registry_stage}/{cand_id}.pt"))
                class_names = cand.get("class_names", [])

                run_dir = out_root / policy_name / registry_stage / cand_id
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "train.log"

                cmd = [
                    sys.executable, "-m", "bloodcancer.train",
                    "--data_dir", str(data_dir),
                    "--model_name", model_name,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--img_size", str(args.img_size),
                    "--num_workers", str(args.num_workers),
                    "--out_dir", str(run_dir),
                ]
                rc = _run(cmd, log_path)
                if rc != 0:
                    print(f"[FAIL] {policy_name}/{registry_stage}/{cand_id} (see {log_path})")
                    continue

                # copy best checkpoint
                best_src = run_dir / "models" / "best.pt"
                if not best_src.exists():
                    print(f"[WARN] best checkpoint not found at {best_src}. Skipping registry.")
                    continue
                ckpt_target.parent.mkdir(parents=True, exist_ok=True)
                ckpt_target.write_bytes(best_src.read_bytes())

                # read history.json for metrics
                metrics: dict = {}
                hist_path = run_dir / "history.json"
                if hist_path.exists():
                    try:
                        hist = json.loads(hist_path.read_text(encoding="utf-8"))
                        if hist:
                            last = hist[-1]
                            val = (last.get("val") or {})
                            metrics = {
                                "best_accuracy": last.get("best_accuracy") or val.get("accuracy"),
                                "val": val,
                                "train": last.get("train") or {},
                                "per_class": val.get("per_class") or {},
                                "confusion_matrix": val.get("confusion_matrix"),
                            }
                    except Exception:
                        metrics = {}

                register_checkpoint(ModelEntry(
                    id=cand_id,
                    policy=policy_name,
                    stage=registry_stage,
                    model_name=model_name,
                    checkpoint=str(ckpt_target),
                    class_names=list(class_names) if class_names else [],
                    metrics=metrics,
                    created_at=utc_iso(),
                ))

                print(f"[OK] Trained + registered {policy_name}/{registry_stage}/{cand_id} -> {ckpt_target}")

if __name__ == "__main__":
    main()
