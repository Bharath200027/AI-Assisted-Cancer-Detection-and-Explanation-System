from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional, Dict, List
import datetime as dt
import uuid

DEFAULT_DB = Path("artifacts/db.sqlite3")

BASE_SCHEMA = """CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename TEXT,
    image_path TEXT NOT NULL,
    heatmap_path TEXT,
    predicted_label TEXT,
    confidence REAL,
    probs_json TEXT,
    needs_human_review INTEGER DEFAULT 1,
    human_label TEXT,
    notes TEXT
);"""

JOBS_SCHEMA = """CREATE TABLE IF NOT EXISTS train_jobs (
            kind TEXT DEFAULT 'incremental',
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    log_path TEXT,
    checkpoint_path TEXT,
    metrics_json TEXT
);"""

EXTRA_COLUMNS = {
    "trace_json": "TEXT",
    "active_learning_score": "REAL",
    "suggested_for_labeling": "INTEGER DEFAULT 0",
    "policy": "TEXT",
    "selected_model_name": "TEXT",
    "selected_checkpoint": "TEXT",
    "mode": "TEXT",
    "disagreement": "REAL",
    "predicted_subtype": "TEXT",
    "subtype_confidence": "REAL",
    "subtype_probs_json": "TEXT",
    "subtype_disagreement": "REAL",
}

def _conn(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(BASE_SCHEMA)
    conn.execute(JOBS_SCHEMA)
    conn.commit()
    # best-effort migration: add missing columns on existing db
    try:
        cols = {r[1] for r in conn.execute('PRAGMA table_info(train_jobs)').fetchall()}
        if 'kind' not in cols:
            conn.execute("ALTER TABLE train_jobs ADD COLUMN kind TEXT DEFAULT 'incremental'")
            conn.commit()
    except Exception:
        pass
    _ensure_case_columns(conn)
    return conn

def _ensure_case_columns(conn: sqlite3.Connection) -> None:
    # Lightweight migration: add missing columns
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(cases)").fetchall()}
    for col, ddl in EXTRA_COLUMNS.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE cases ADD COLUMN {col} {ddl}")
    conn.commit()

CONN = _conn()

def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def create_case(*, filename: str, image_path: str) -> str:
    case_id = uuid.uuid4().hex
    CONN.execute(
        "INSERT INTO cases (id, created_at, filename, image_path, needs_human_review, suggested_for_labeling) VALUES (?,?,?,?,1,0)",
        (case_id, now_iso(), filename, image_path),
    )
    CONN.commit()
    return case_id

def update_case(case_id: str, fields: Dict[str, Any]) -> None:
    keys = list(fields.keys())
    if not keys:
        return
    sql = "UPDATE cases SET " + ", ".join([f"{k}=?" for k in keys]) + " WHERE id=?"
    vals = [fields[k] for k in keys] + [case_id]
    CONN.execute(sql, vals)
    CONN.commit()

def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    row = CONN.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return dict(row) if row else None

def list_cases(limit: int = 200, predicted_label: Optional[str]=None, predicted_subtype: Optional[str]=None, needs_review: Optional[bool]=None,
               suggested_for_labeling: Optional[bool]=None) -> List[Dict[str, Any]]:
    where = []
    params: list[Any] = []
    if predicted_label:
        where.append("predicted_label=?")
        params.append(predicted_label)
    if predicted_subtype:
        where.append("predicted_subtype=?")
        params.append(predicted_subtype)
    if needs_review is not None:
        where.append("needs_human_review=?")
        params.append(1 if needs_review else 0)
    if suggested_for_labeling is not None:
        where.append("suggested_for_labeling=?")
        params.append(1 if suggested_for_labeling else 0)

    wh = (" WHERE " + " AND ".join(where)) if where else ""
    rows = CONN.execute(f"SELECT * FROM cases{wh} ORDER BY created_at DESC LIMIT ?", (*params, limit)).fetchall()
    return [dict(r) for r in rows]

def set_human_label(case_id: str, label: str, notes: str="") -> None:
    CONN.execute(
        "UPDATE cases SET human_label=?, notes=? WHERE id=?",
        (label, notes, case_id),
    )
    CONN.commit()

def create_job(status: str="queued", kind: str="incremental", log_path: str="", checkpoint_path: str="", metrics: Optional[dict]=None) -> str:
    job_id = uuid.uuid4().hex
    CONN.execute(
        "INSERT INTO train_jobs (id, created_at, status, kind, log_path, checkpoint_path, metrics_json) VALUES (?,?,?,?,?,?,?)",
        (job_id, now_iso(), status, kind, log_path, checkpoint_path, json.dumps(metrics or {})),
    )
    CONN.commit()
    return job_id

def update_job(job_id: str, **fields) -> None:
    if "metrics" in fields:
        fields["metrics_json"] = json.dumps(fields.pop("metrics") or {})
    keys = list(fields.keys())
    if not keys:
        return
    sql = "UPDATE train_jobs SET " + ", ".join([f"{k}=?" for k in keys]) + " WHERE id=?"
    vals = [fields[k] for k in keys] + [job_id]
    CONN.execute(sql, vals)
    CONN.commit()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    row = CONN.execute("SELECT * FROM train_jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None

def list_jobs(limit: int=50) -> List[Dict[str, Any]]:
    rows = CONN.execute("SELECT * FROM train_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

def stats() -> Dict[str, Any]:
    rows = CONN.execute(
        "SELECT predicted_label, COUNT(*) as n FROM cases WHERE predicted_label IS NOT NULL GROUP BY predicted_label"
    ).fetchall()
    counts = {r["predicted_label"]: int(r["n"]) for r in rows}

    srows = CONN.execute(
        "SELECT predicted_subtype, COUNT(*) as n FROM cases WHERE predicted_subtype IS NOT NULL GROUP BY predicted_subtype"
    ).fetchall()
    subtype_counts = {r["predicted_subtype"]: int(r["n"]) for r in srows if r["predicted_subtype"]}

    total = int(CONN.execute("SELECT COUNT(*) as n FROM cases").fetchone()["n"])
    needs_review = int(CONN.execute("SELECT COUNT(*) as n FROM cases WHERE needs_human_review=1").fetchone()["n"])
    suggested = int(CONN.execute("SELECT COUNT(*) as n FROM cases WHERE suggested_for_labeling=1").fetchone()["n"])
    labeled = int(CONN.execute("SELECT COUNT(*) as n FROM cases WHERE human_label IS NOT NULL").fetchone()["n"])

    acc = None
    if labeled > 0:
        good = int(CONN.execute(
            "SELECT COUNT(*) as n FROM cases WHERE human_label IS NOT NULL AND predicted_label = human_label"
        ).fetchone()["n"])
        acc = good / labeled

    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]
    hist = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": 0 for i in range(len(bins)-1)}
    conf_rows = CONN.execute("SELECT confidence FROM cases WHERE confidence IS NOT NULL").fetchall()
    for r in conf_rows:
        c = float(r["confidence"])
        for i in range(len(bins)-1):
            if bins[i] <= c < bins[i+1]:
                hist[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] += 1
                break

    return {
        "total_cases": total,
        "needs_human_review": needs_review,
        "suggested_for_labeling": suggested,
        "labeled_cases": labeled,
        "label_counts": counts,
        "subtype_counts": subtype_counts,
        "labeled_accuracy": acc,
        "confidence_histogram": hist,
    }
