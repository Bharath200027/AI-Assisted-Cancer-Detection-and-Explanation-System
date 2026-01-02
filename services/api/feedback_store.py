from __future__ import annotations
from pathlib import Path
import shutil
import uuid
import json
import datetime as dt

FEEDBACK_ROOT = Path("data/feedback")
META_FILE = FEEDBACK_ROOT / "meta.jsonl"

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def save_feedback(image_path: str, label: str, case_id: str, notes: str="") -> str:
    """Copy the case image into data/feedback/<label>/ and append metadata."""
    src = Path(image_path)
    FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)
    dest_dir = FEEDBACK_ROOT / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    ext = src.suffix.lower() or ".png"
    dest = dest_dir / f"{case_id}_{uuid.uuid4().hex}{ext}"
    shutil.copy2(src, dest)

    rec = {
        "ts": now_iso(),
        "case_id": case_id,
        "label": label,
        "path": str(dest),
        "notes": notes,
    }
    with META_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return str(dest)
