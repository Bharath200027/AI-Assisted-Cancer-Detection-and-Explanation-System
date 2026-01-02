from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import datetime as dt

DEFAULT_REGISTRY_PATH = Path("artifacts/model_registry.json")

def utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

@dataclass
class ModelEntry:
    """Registry record for a trained checkpoint.

    metrics: expected to include keys such as:
      - best_accuracy
      - val: {accuracy, f1_macro, auc?, confusion_matrix, per_class}
      - per_class: {<class_name>: {precision, recall, f1, support}}
    """
    id: str
    policy: str
    stage: str
    model_name: str
    checkpoint: str
    class_names: List[str]
    metrics: Dict[str, Any]
    created_at: str

def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "updated_at": utc_iso(), "entries": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "updated_at": utc_iso(), "entries": []}

def save_registry(reg: Dict[str, Any], path: Path = DEFAULT_REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reg["updated_at"] = utc_iso()
    path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

def register_checkpoint(entry: ModelEntry, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    reg = load_registry(path)
    entries = reg.get("entries", [])
    entries = [e for e in entries if not (e.get("id")==entry.id and e.get("checkpoint")==entry.checkpoint)]
    entries.append(asdict(entry))
    reg["entries"] = entries
    save_registry(reg, path)

def iter_entries(policy: Optional[str]=None, stage: Optional[str]=None, model_id: Optional[str]=None, path: Path = DEFAULT_REGISTRY_PATH):
    reg = load_registry(path)
    for e in reg.get("entries", []):
        if policy and e.get("policy") != policy: 
            continue
        if stage and e.get("stage") != stage:
            continue
        if model_id and e.get("id") != model_id:
            continue
        yield e

def _best_accuracy(e: dict) -> float:
    m = e.get("metrics") or {}
    return float(m.get("best_accuracy") or (m.get("val") or {}).get("accuracy") or m.get("accuracy") or 0.0)

def best_checkpoint(policy: str, stage: str, model_id: Optional[str]=None, path: Path = DEFAULT_REGISTRY_PATH) -> Optional[dict]:
    entries = list(iter_entries(policy=policy, stage=stage, model_id=model_id, path=path))
    if not entries:
        return None
    entries.sort(key=lambda e: (_best_accuracy(e), e.get("created_at","")), reverse=True)
    return entries[0]

def per_class_score(e: dict, class_name: str, metric: str = "f1") -> float:
    m = e.get("metrics") or {}
    pc = m.get("per_class") or (m.get("val") or {}).get("per_class") or {}
    if class_name in pc and isinstance(pc[class_name], dict):
        return float(pc[class_name].get(metric) or 0.0)
    return 0.0

def best_checkpoint_for_class(
    policy: str,
    stage: str,
    class_name: str,
    model_ids: Optional[List[str]] = None,
    metric: str = "f1",
    path: Path = DEFAULT_REGISTRY_PATH,
) -> Optional[dict]:
    entries = list(iter_entries(policy=policy, stage=stage, path=path))
    if model_ids:
        entries = [e for e in entries if e.get("id") in set(model_ids)]
    if not entries:
        return None
    entries.sort(key=lambda e: (per_class_score(e, class_name, metric=metric), _best_accuracy(e), e.get("created_at","")), reverse=True)
    best = entries[0]
    if per_class_score(best, class_name, metric=metric) <= 0.0:
        # fall back to best overall if per-class unavailable
        return best_checkpoint(policy, stage, model_id=(best.get("id")), path=path)
    return best

def best_models_per_class(
    policy: str,
    stage: str,
    class_names: List[str],
    candidates: List[dict],
    metric: str = "f1",
    path: Path = DEFAULT_REGISTRY_PATH,
) -> Tuple[Dict[str, dict], List[dict]]:
    """Return mapping class->best entry (restricted to provided candidates) and list of unique selected candidate dicts."""
    ids = [c.get("id") for c in candidates if c.get("id")]
    selected_map: Dict[str, dict] = {}
    selected_ids = set()
    selected_candidates: List[dict] = []

    for cn in class_names:
        best = best_checkpoint_for_class(policy, stage, cn, model_ids=ids or None, metric=metric, path=path)
        if best:
            selected_map[cn] = best
            selected_ids.add(best.get("id"))
    # Build unique candidate dicts for inference
    for c in candidates:
        if c.get("id") in selected_ids:
            selected_candidates.append(c)
    if not selected_candidates and candidates:
        selected_candidates = [candidates[0]]
    return selected_map, selected_candidates
