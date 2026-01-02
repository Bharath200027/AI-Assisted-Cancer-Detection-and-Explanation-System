from __future__ import annotations
import argparse
from bloodcancer.registry import register_checkpoint, ModelEntry, utc_iso

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--class_names", nargs="+", required=True)
    ap.add_argument("--best_accuracy", type=float, default=0.0)
    args = ap.parse_args()

    register_checkpoint(ModelEntry(
        id=args.id,
        policy=args.policy,
        stage=args.stage,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        class_names=list(args.class_names),
        metrics={"best_accuracy": args.best_accuracy},
        created_at=utc_iso(),
    ))
    print("Registered:", args.checkpoint)

if __name__ == "__main__":
    main()
