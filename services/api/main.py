from __future__ import annotations
import os
from pathlib import Path
import shutil
import uuid
import subprocess
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bloodcancer.agents.graph import create_graph
from services.api.db import create_case, update_case, get_case, list_cases, stats as get_stats, create_job, update_job, get_job, list_jobs
from services.api.feedback_store import save_feedback

app = FastAPI(title="Blood Cancer AI API (Research)", version="0.5.0")

GRAPH = None

ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
UPLOADS = ARTIFACTS / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

# Expose artifacts for UI (images/heatmaps)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS)), name="artifacts")

@app.on_event("startup")
def _startup():
    global GRAPH
    GRAPH = create_graph(os.getenv("APP_CONFIG", "configs/app.yaml"))

@app.get("/health")
def health():
    return {"status": "ok"}

class LabelRequest(BaseModel):
    human_label: str
    notes: str = ""

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    ext = Path(file.filename).suffix.lower() or ".png"
    case_id = create_case(filename=file.filename, image_path="")
    img_path = UPLOADS / f"{case_id}{ext}"

    with img_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    update_case(case_id, {"image_path": str(img_path)})

    state = {"image_path": str(img_path)}
    try:
        result = GRAPH.invoke(state)
    except Exception as e:
        update_case(case_id, {"notes": f"pipeline_error: {e}"})
        return JSONResponse(status_code=500, content={"error": str(e), "case_id": case_id})

    # Persist inference results
    update_fields = {
        "heatmap_path": result.get("heatmap_path"),
        "predicted_label": result.get("predicted_label"),
        "confidence": result.get("confidence"),
        "probs_json": __import__("json").dumps(result.get("probs") or {}),
        "needs_human_review": 1 if result.get("needs_human_review", True) else 0,
        "trace_json": __import__("json").dumps({"plan": result.get("plan"), "tool_trace": result.get("tool_trace"), "stage1_snapshot": result.get("stage1_snapshot"), "subtype": {"predicted_subtype": result.get("predicted_subtype"), "subtype_confidence": result.get("subtype_confidence"), "subtype_probs_json": __import__("json").dumps(result.get("subtype_probs") or {}), "subtype_disagreement": result.get("subtype_disagreement")}}),
        "predicted_subtype": result.get("predicted_subtype"),
        "subtype_confidence": result.get("subtype_confidence"),
        "subtype_probs_json": __import__("json").dumps(result.get("subtype_probs") or {}),
        "subtype_disagreement": result.get("subtype_disagreement"),
        "active_learning_score": result.get("active_learning_score"),
        "suggested_for_labeling": 1 if result.get("suggested_for_labeling", False) else 0,
        "policy": result.get("policy"),
        "mode": result.get("mode"),
        "disagreement": result.get("disagreement"),
        "selected_model_name": result.get("selected_model_name"),
        "selected_checkpoint": result.get("selected_checkpoint"),
    }
    update_case(case_id, update_fields)

    return {
        "case_id": case_id,
        "predicted_label": result.get("predicted_label"),
        "confidence": result.get("confidence"),
        "probs": result.get("probs"),
        "disagreement": result.get("disagreement"),
        "policy": result.get("policy"),
        "mode": result.get("mode"),
        "disagreement": result.get("disagreement"),
        "mode": result.get("mode"),
        "selected_model_name": result.get("selected_model_name"),
        "selected_checkpoint": result.get("selected_checkpoint"),
        "predicted_subtype": result.get("predicted_subtype"),
        "subtype_confidence": result.get("subtype_confidence"),
        "subtype_probs_json": __import__("json").dumps(result.get("subtype_probs") or {}),
        "subtype_disagreement": result.get("subtype_disagreement"),
        "active_learning_score": result.get("active_learning_score"),
        "suggested_for_labeling": result.get("suggested_for_labeling"),
        "heatmap_path": result.get("heatmap_path"),
        "explain_summary": result.get("explain_summary"),
        "needs_human_review": result.get("needs_human_review", True),
        "report_text": result.get("report_text", ""),
        "errors": result.get("errors", []),
        # Helpful URLs for UI
        "image_url": f"/artifacts/uploads/{img_path.name}",
        "heatmap_url": f"/artifacts/{Path(result.get('heatmap_path','')).relative_to(ARTIFACTS) if result.get('heatmap_path') else ''}",
    }

@app.get("/cases")
def cases(limit: int = 200, predicted_label: str | None = None, predicted_subtype: str | None = None, needs_review: bool | None = None, suggested_for_labeling: bool | None = None):
    return {"items": list_cases(limit=limit, predicted_label=predicted_label, predicted_subtype=predicted_subtype, needs_review=needs_review, suggested_for_labeling=suggested_for_labeling)}

@app.get("/cases/{case_id}")
def case(case_id: str):
    c = get_case(case_id)
    if not c:
        raise HTTPException(status_code=404, detail="case not found")
    # decode probs_json
    if c.get("probs_json"):
        try:
            c["probs"] = __import__("json").loads(c["probs_json"])
        except Exception:
            c["probs"] = {}
    if c.get("subtype_probs_json"):
        try:
            c["subtype_probs"] = __import__("json").loads(c["subtype_probs_json"])
        except Exception:
            c["subtype_probs"] = {}
    if c.get("trace_json"):
        try:
            c["trace"] = __import__("json").loads(c["trace_json"])
        except Exception:
            c["trace"] = {}
    return c

@app.post("/cases/{case_id}/label")
def label_case(case_id: str, req: LabelRequest):
    c = get_case(case_id)
    if not c:
        raise HTTPException(status_code=404, detail="case not found")
    # update label in db
    from services.api.db import set_human_label
    set_human_label(case_id, req.human_label, req.notes)

    # copy into feedback folder for continual improvement
    save_feedback(c["image_path"], req.human_label, case_id, req.notes)

    # Optional: auto-retrain after enough new labels (safe, audit-friendly)
    try:
        from bloodcancer.config import load_yaml
        cl = load_yaml("configs/continual.yaml").get("continual_learning", {})
        if bool(cl.get("auto_retrain", False)):
            # Count how many feedback examples exist
            from pathlib import Path
            fb_root = Path("data/feedback")
            n_fb = sum(1 for p in fb_root.rglob("*") if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"})
            min_n = int(cl.get("auto_retrain_min_new_labels", 50))
            if n_fb >= min_n:
                # fire and forget training job
                payload = TrainRequest(
                    base_data_dir="data/processed/cnmc",
                    feedback_dir="data/feedback",
                    out_dir="artifacts",
                    epochs=int(cl.get("auto_retrain_epochs", 3)),
                    model_name=str(cl.get("auto_retrain_model_name", "tf_efficientnetv2_s")),
                    img_size=224,
                    batch_size=32,
                )
                trigger_train(payload)
    except Exception:
        pass

    return {"ok": True}

@app.get("/stats")
def stats():
    return get_stats()


@app.get("/agents")
def agents():
    # Return agent specs (propositions, IO, guardrails)
    from bloodcancer.config import AppConfig, load_yaml
    from bloodcancer.agents.agents import list_agent_specs
    from bloodcancer.agents.tools import list_tool_specs
    cfg = AppConfig(load_yaml(os.getenv("APP_CONFIG", "configs/app.yaml")))
    return {"agents": list_agent_specs(cfg)}

@app.get("/tools")
def tools():
    from bloodcancer.config import AppConfig, load_yaml
    from bloodcancer.agents.tools import list_tool_specs
    cfg = AppConfig(load_yaml(os.getenv("APP_CONFIG", "configs/app.yaml")))
    return {"tools": list_tool_specs(cfg)}


@app.get("/train/jobs")
def train_jobs(limit: int=50):
    return {"items": list_jobs(limit=limit)}

def _run_training_job(job_id: str, base_data_dir: str, feedback_dir: str, out_dir: str, epochs: int, model_name: str, img_size: int, batch_size: int):
    log_dir = Path(out_dir) / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    update_job(job_id, status="running", log_path=str(log_path))

    cmd = [
        "python", "-m", "bloodcancer.train_incremental",
        "--base_data_dir", base_data_dir,
        "--feedback_dir", feedback_dir,
        "--out_dir", out_dir,
        "--epochs", str(epochs),
        "--model_name", model_name,
        "--img_size", str(img_size),
        "--batch_size", str(batch_size),
    ]
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Command: " + " ".join(cmd) + "\n\n")
        f.flush()
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
            # best checkpoint is written by trainer to artifacts/models/best.pt
            ckpt = str(Path(out_dir) / "models" / "best.pt")
            status = "succeeded" if proc.returncode == 0 else "failed"
            update_job(job_id, status=status, checkpoint_path=ckpt)
        except Exception as e:
            update_job(job_id, status="failed")
            f.write(f"Exception: {e}\n")

class TrainRequest(BaseModel):
    base_data_dir: str = "data/processed/cnmc"
    feedback_dir: str = "data/feedback"
    out_dir: str = "artifacts"
    epochs: int = 3
    model_name: str = "tf_efficientnetv2_s"
    img_size: int = 224
    batch_size: int = 32

class TrainPoliciesRequest(BaseModel):
    training_cfg: str = "configs/training.yaml"
    models_cfg: str = "configs/models.yaml"
    out_root: str = "artifacts/policy_runs"
    epochs: int = 10
    batch_size: int = 32
    img_size: int = 224



@app.post("/train/trigger")
def trigger_train(req: TrainRequest):
    # Create job and start in a background thread
    job_id = create_job(status="queued")
    t = threading.Thread(
        target=_run_training_job,
        args=(job_id, req.base_data_dir, req.feedback_dir, req.out_dir, req.epochs, req.model_name, req.img_size, req.batch_size),
        daemon=True
    )
    t.start()
    return {"job_id": job_id, "status": "queued", "note": "Training started in background. Poll /train/jobs or /train/job/{id}."}


@app.post("/train/policies_trigger")
def train_policies_trigger(req: TrainPoliciesRequest):
    """Train multiple policy/stage candidates defined in configs/models.yaml using configs/training.yaml.

    Runs asynchronously in a background thread and registers checkpoints into artifacts/model_registry.json.
    """
    job_id = create_job(status="queued", kind="policies")

    def _run():
        update_job(job_id, status="running")
        log_path = Path("artifacts") / "jobs" / f"{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            os.getenv("PYTHON", "python"),
            "scripts/train_policies.py",
            "--training_cfg", req.training_cfg,
            "--models_cfg", req.models_cfg,
            "--out_root", req.out_root,
            "--epochs", str(req.epochs),
            "--batch_size", str(req.batch_size),
            "--img_size", str(req.img_size),
        ]
        with log_path.open("w", encoding="utf-8") as f:
            f.write("Command: " + " ".join(cmd) + "\n\n")
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
            status = "succeeded" if proc.returncode == 0 else "failed"
            update_job(job_id, status=status, log_path=str(log_path))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued", "note": "Policy training started in background. Poll /train/jobs or /train/job/{id}."}

@app.get("/train/job/{job_id}")
def train_job(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    # decode metrics_json
    if j.get("metrics_json"):
        try:
            j["metrics"] = __import__("json").loads(j["metrics_json"])
        except Exception:
            j["metrics"] = {}
    return j
