from __future__ import annotations

import os
import io
import time
from typing import Any, Dict, Optional
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Blood Cancer AI Dashboard (Research)", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# -----------------------------
# API Client (small + robust)
# -----------------------------
def api_request(method: str, path: str, *, params: dict | None = None, files: dict | None = None, json_body: dict | None = None, timeout: int = 120) -> dict:
    url = f"{API_URL}{path}"
    try:
        r = requests.request(method, url, params=params, files=files, json=json_body, timeout=timeout)
        if r.status_code >= 400:
            return {"_error": f"{r.status_code} {r.reason}", "_detail": (r.text[:2000] if r.text else "")}
        return r.json() if r.content else {}
    except requests.RequestException as e:
        return {"_error": "request_failed", "_detail": str(e)}

def api_get(path: str, params: dict | None = None) -> dict:
    return api_request("GET", path, params=params)

def api_post(path: str, *, json_body: dict | None = None, files: dict | None = None) -> dict:
    return api_request("POST", path, json_body=json_body, files=files)

def show_error(resp: dict):
    if resp.get("_error"):
        st.error(f"API error: {resp.get('_error')}")
        if resp.get("_detail"):
            st.code(resp.get("_detail"))
        return True
    return False

# -----------------------------
# UI
# -----------------------------
st.title("AI-Assisted Blood Cancer Detection & Explanation (Research)")
st.caption("Upload images, inspect predictions + subtypes, review flagged cases, manage active-learning, and trigger retraining.")

with st.sidebar:
    st.subheader("Connection")
    st.write(f"API: `{API_URL}`")
    if st.button("Ping /health"):
        h = api_get("/health")
        if show_error(h):
            st.stop()
        st.success(h.get("status", "ok"))

tabs = st.tabs([
    "Upload & Predict",
    "Buckets (By Type/Subtype)",
    "Cases Table",
    "Stats",
    "Review & Label",
    "Active Learning",
    "Retrain / Continual Learning",
    "Agents",
    "Tools",
])

# -----------------------------
# Tab 0: Upload & Predict
# -----------------------------
with tabs[0]:
    st.subheader("Upload & Predict")
    uploaded = st.file_uploader("Upload blood smear image(s)", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
    run = st.button("Run prediction", disabled=not uploaded)

    if run and uploaded:
        for up in uploaded:
            st.markdown("---")
            st.write(f"### {up.name}")
            st.image(up, caption="Uploaded", use_container_width=True)

            files = {"file": (up.name, up.getvalue(), up.type or "application/octet-stream")}
            res = api_post("/predict", files=files)
            if show_error(res):
                continue

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Predicted", res.get("predicted_label", "n/a"))
            col2.metric("Confidence", f"{float(res.get('confidence', 0.0)):.3f}")
            col3.metric("Needs Review", str(bool(res.get("needs_human_review", True))))
            col4.metric("Case ID", (res.get("case_id","")[:12] if res.get("case_id") else "n/a"))

            if res.get("predicted_subtype"):
                st.info(f"Subtype: **{res.get('predicted_subtype')}** (conf={float(res.get('subtype_confidence',0.0)):.3f}, mode={res.get('subtype_mode','n/a')})")

            # Prob tables
            cA, cB = st.columns(2)
            with cA:
                st.markdown("#### Stage-1 probabilities")
                probs = res.get("probs") or {}
                if probs:
                    dfp = pd.DataFrame([probs]).T.rename(columns={0:"prob"}).sort_values("prob", ascending=False)
                    st.dataframe(dfp, use_container_width=True, height=200)
            with cB:
                st.markdown("#### Stage-2 probabilities (if any)")
                sp = res.get("subtype_probs") or {}
                if sp:
                    dfs = pd.DataFrame([sp]).T.rename(columns={0:"prob"}).sort_values("prob", ascending=False)
                    st.dataframe(dfs, use_container_width=True, height=200)
                else:
                    st.caption("No stage-2 output for this case.")

            # Heatmap + report
            if res.get("heatmap_url"):
                st.markdown("#### Explainability (Grad-CAM overlay)")
                st.image(f"{API_URL}{res['heatmap_url']}", use_container_width=True)

            if res.get("report_text"):
                st.markdown("#### Report")
                st.code(res.get("report_text",""), language="markdown")

            with st.expander("Execution trace"):
                st.json(res.get("tool_trace", []))

# -----------------------------
# Tab 1: Buckets
# -----------------------------
with tabs[1]:
    st.subheader("Buckets")
    st.caption("Browse images grouped by predicted type and optional subtype.")

    stats = api_get("/stats")
    if show_error(stats):
        st.stop()

    label_counts = stats.get("label_counts", {}) or {}
    subtype_counts = stats.get("subtype_counts", {}) or {}

    c1, c2 = st.columns(2)
    with c1:
        chosen_label = st.selectbox("Predicted type", ["(all)"] + sorted(label_counts.keys()))
    with c2:
        chosen_sub = st.selectbox("Subtype (optional)", ["(all)"] + sorted(subtype_counts.keys()))

    params: Dict[str, Any] = {"limit": 60}
    if chosen_label != "(all)":
        params["predicted_label"] = chosen_label
    if chosen_sub != "(all)":
        params["predicted_subtype"] = chosen_sub

    items = api_get("/cases", params=params).get("items", [])
    if not items:
        st.info("No cases match the selected filters.")
    else:
        st.write(f"Showing {len(items)} most recent cases.")
        cols = st.columns(4)
        for i, c in enumerate(items):
            with cols[i % 4]:
                img_url = c.get("image_path")
                if img_url:
                    # stored as local path; API serves via /artifacts
                    name = Path(img_url).name
                    st.image(f"{API_URL}/artifacts/uploads/{name}", use_container_width=True)
                st.caption(f"{c.get('predicted_label','n/a')} • {c.get('predicted_subtype','-')} • conf={float(c.get('confidence',0.0)):.2f}")
                st.caption(c.get("id","")[:12])

# -----------------------------
# Tab 2: Cases Table
# -----------------------------
with tabs[2]:
    st.subheader("Cases Table")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        label_filter = st.text_input("Filter predicted_label (exact)", "")
    with f2:
        subtype_filter = st.text_input("Filter predicted_subtype (exact)", "")
    with f3:
        needs_review = st.selectbox("Needs review", ["(any)","yes","no"])
    with f4:
        limit = st.slider("Limit", 50, 500, 200, step=50)

    params: Dict[str, Any] = {"limit": int(limit)}
    if label_filter.strip():
        params["predicted_label"] = label_filter.strip()
    if subtype_filter.strip():
        params["predicted_subtype"] = subtype_filter.strip()
    if needs_review == "yes":
        params["needs_review"] = True
    elif needs_review == "no":
        params["needs_review"] = False

    data = api_get("/cases", params=params)
    if show_error(data):
        st.stop()
    items = data.get("items", [])
    if not items:
        st.info("No cases found.")
    else:
        df = pd.DataFrame(items)
        show_cols = [c for c in [
            "id","created_at","filename","predicted_label","predicted_subtype","confidence",
            "needs_human_review","suggested_for_labeling","active_learning_score","human_label","notes"
        ] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, height=420)

        with st.expander("Inspect a case"):
            case_id = st.selectbox("Case ID", df["id"].tolist())
            c = api_get(f"/cases/{case_id}")
            if not show_error(c):
                left, right = st.columns([1,1])
                with left:
                    img_path = c.get("image_path","")
                    if img_path:
                        st.image(f"{API_URL}/artifacts/uploads/{Path(img_path).name}", use_container_width=True)
                    if c.get("heatmap_path"):
                        st.image(f"{API_URL}/artifacts/heatmaps/{Path(c['heatmap_path']).name}", caption="Heatmap", use_container_width=True)
                with right:
                    st.json({k: c.get(k) for k in ["predicted_label","predicted_subtype","confidence","needs_human_review","human_label","notes"]})
                    st.markdown("**Probabilities**")
                    st.json(c.get("probs", {}))
                    if c.get("subtype_probs"):
                        st.markdown("**Subtype probs**")
                        st.json(c.get("subtype_probs", {}))
                    st.markdown("**Trace**")
                    st.json(c.get("trace", {}))

# -----------------------------
# Tab 3: Stats
# -----------------------------
with tabs[3]:
    st.subheader("Stats")
    stats = api_get("/stats")
    if show_error(stats):
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cases", int(stats.get("total", 0)))
    col2.metric("Needs review", int(stats.get("needs_review", 0)))
    col3.metric("Feedback labels", int(stats.get("feedback_labels", 0)))

    st.markdown("### Predicted label distribution")
    label_counts = stats.get("label_counts", {}) or {}
    if label_counts:
        st.bar_chart(pd.Series(label_counts).sort_values(ascending=False))
    else:
        st.info("No label counts yet.")

    st.markdown("### Subtype distribution")
    subtype_counts = stats.get("subtype_counts", {}) or {}
    if subtype_counts:
        st.bar_chart(pd.Series(subtype_counts).sort_values(ascending=False))
    else:
        st.caption("No subtype counts yet.")

    st.markdown("### Confidence histogram (recent)")
    hist = stats.get("confidence_hist", {}) or {}
    if hist:
        st.bar_chart(pd.Series(hist))
    else:
        st.caption("No confidence histogram yet.")

# -----------------------------
# Tab 4: Review & Label
# -----------------------------
with tabs[4]:
    st.subheader("Review & Label (Human-in-the-loop)")
    st.caption("Label cases flagged for review. These labels feed incremental training.")

    items = api_get("/cases", params={"limit": 200, "needs_review": True}).get("items", [])
    if not items:
        st.info("No cases flagged for review.")
    else:
        ids = [c["id"] for c in items if c.get("id")]
        selected = st.selectbox("Select case", ids, format_func=lambda x: x[:12])

        case = api_get(f"/cases/{selected}")
        if show_error(case):
            st.stop()

        st.write(f"**Predicted:** {case.get('predicted_label')} (conf={float(case.get('confidence',0.0)):.3f})")
        if case.get("predicted_subtype"):
            st.write(f"**Subtype:** {case.get('predicted_subtype')} (conf={float(case.get('subtype_confidence',0.0)):.3f})")

        img_path = case.get("image_path","")
        if img_path:
            st.image(f"{API_URL}/artifacts/uploads/{Path(img_path).name}", use_container_width=True)

        label = st.text_input("Human label (use dataset class names)", value=case.get("human_label") or "")
        notes = st.text_area("Notes", value=case.get("notes") or "", height=120)
        if st.button("Submit label"):
            res = api_post(f"/cases/{selected}/label", json_body={"human_label": label, "notes": notes})
            if not show_error(res):
                st.success("Saved feedback label.")

# -----------------------------
# Tab 5: Active Learning
# -----------------------------
with tabs[5]:
    st.subheader("Active Learning Queue")
    st.caption("Cases automatically suggested for labeling (high uncertainty / disagreement).")

    items = api_get("/cases", params={"limit": 200, "suggested_for_labeling": True}).get("items", [])
    if not items:
        st.info("No active-learning suggestions yet. Upload more images to populate the queue.")
    else:
        df = pd.DataFrame(items)
        sort_col = "active_learning_score" if "active_learning_score" in df.columns else "confidence"
        df = df.sort_values(sort_col, ascending=False)
        st.dataframe(df[[c for c in ["id","created_at","predicted_label","predicted_subtype","confidence","disagreement","active_learning_score","needs_human_review"] if c in df.columns]],
                     use_container_width=True, height=380)

        pick = st.selectbox("Pick a case to label", df["id"].tolist(), format_func=lambda x: x[:12])
        case = api_get(f"/cases/{pick}")
        if not show_error(case):
            img_path = case.get("image_path","")
            if img_path:
                st.image(f"{API_URL}/artifacts/uploads/{Path(img_path).name}", use_container_width=True)
            st.write(f"Predicted: {case.get('predicted_label')} / subtype: {case.get('predicted_subtype','-')}")
            label = st.text_input("Human label", key="al_label")
            notes = st.text_area("Notes", key="al_notes", height=100)
            if st.button("Submit label (Active Learning)"):
                res = api_post(f"/cases/{pick}/label", json_body={"human_label": label, "notes": notes})
                if not show_error(res):
                    st.success("Saved label. Consider retraining when you have enough new labels.")

# -----------------------------
# Tab 6: Retrain / Continual
# -----------------------------
with tabs[6]:
    st.subheader("Retrain / Continual Learning")
    st.caption("Trigger incremental fine-tuning from feedback data, or train/refresh policy models.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Incremental retrain (feedback)")
        base_data_dir = st.text_input("Base dataset dir", value="data/processed/cnmc")
        feedback_dir = st.text_input("Feedback dir", value="data/feedback")
        epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3, step=1)
        if st.button("Trigger incremental retrain"):
            resp = api_post("/train/trigger", json_body={
                "base_data_dir": base_data_dir,
                "feedback_dir": feedback_dir,
                "out_dir": "artifacts",
                "epochs": int(epochs),
                "model_name": "tf_efficientnetv2_s",
                "img_size": 224,
                "batch_size": 32
            })
            if not show_error(resp):
                st.success(f"Job created: {resp.get('job_id','n/a')}")
    with colB:
        st.markdown("### Policy training (multi-model)")
        t_epochs = st.number_input("Policy epochs", min_value=1, max_value=50, value=5, step=1, key="pol_epochs")
        if st.button("Trigger policy training"):
            resp = api_post("/train/policies_trigger", json_body={
                "epochs": int(t_epochs),
                "batch_size": 32,
                "img_size": 224
            })
            if not show_error(resp):
                st.success(f"Policy training job: {resp.get('job_id','n/a')}")
                st.info("Check logs in artifacts/policy_runs and registry in artifacts/model_registry.json")

    st.markdown("---")
    st.markdown("### Training jobs")
    jobs = api_get("/train/jobs").get("items", [])
    if jobs:
        st.dataframe(pd.DataFrame(jobs)[[c for c in ["id","kind","status","created_at","updated_at","log_path","out_dir"] if c in pd.DataFrame(jobs).columns]],
                     use_container_width=True, height=320)
    else:
        st.caption("No jobs yet.")

# -----------------------------
# Tab 7: Agents
# -----------------------------
with tabs[7]:
    st.subheader("Agents")
    data = api_get("/agents")
    if show_error(data):
        st.stop()
    st.json(data.get("agents", []))

# -----------------------------
# Tab 8: Tools
# -----------------------------
with tabs[8]:
    st.subheader("Tools")
    data = api_get("/tools")
    if show_error(data):
        st.stop()
    st.json(data.get("tools", []))
