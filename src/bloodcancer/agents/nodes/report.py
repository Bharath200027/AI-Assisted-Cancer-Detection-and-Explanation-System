# Deprecated: node logic moved to src/bloodcancer/agents/agents.py and src/bloodcancer/agents/nodes.py
from __future__ import annotations
from pathlib import Path

from bloodcancer.reporting import render_report
from bloodcancer.llm import call_llm

def report_node(state: dict, cfg) -> dict:
    if state.get("errors"):
        # produce minimal report
        report = "Errors encountered:\n" + "\n".join(f"- {e}" for e in state["errors"])
        return {"report_text": report, "needs_human_review": True}

    evidences = []
    for e in state.get("evidences", []) or []:
        evidences.append(type("Evidence", (), e))  # lightweight object with attrs

    llm_text = ""
    if cfg.report_use_llm:
        # ground the prompt in structured facts only
        prompt = (
            "Draft a cautious decision-support interpretation for a blood smear image classification.\n"
            f"Predicted label: {state['predicted_label']}\n"
            f"Confidence: {state['confidence']:.3f}\n"
            f"Explainability summary: {state.get('explain_summary','')}\n"
            "Constraints: Not a diagnosis; never prescribe treatment; mention uncertainty; recommend clinician review.\n"
        )
        llm_text = call_llm(prompt)

    report = render_report(
        cfg.report_template,
        image_filename=state.get("image_filename", Path(state["image_path"]).name),
        predicted_label=state["predicted_label"],
        confidence=state["confidence"],
        probs=state["probs"],
        explain_method="Grad-CAM",
        heatmap_path=state.get("heatmap_path", ""),
        explain_summary=state.get("explain_summary", ""),
        evidences=evidences,
        confidence_threshold=cfg.confidence_threshold,
        use_llm=cfg.report_use_llm,
        llm_text=llm_text,
    )

    return {"report_text": report}
