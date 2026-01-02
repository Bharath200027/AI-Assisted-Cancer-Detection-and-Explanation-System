# Deprecated: node logic moved to src/bloodcancer/agents/agents.py and src/bloodcancer/agents/nodes.py
from __future__ import annotations

def safety_node(state: dict, cfg) -> dict:
    if state.get("errors"):
        return {"needs_human_review": True}

    needs_review = bool(state.get("confidence", 0.0) < cfg.confidence_threshold)
    # enforce disclaimer presence
    report = state.get("report_text", "")
    if "Not a diagnosis" not in report:
        report += "\n\n> **Not a diagnosis.** This output is for research/decision-support only and must be reviewed by a qualified clinician.\n"
    if needs_review and "human review" not in report.lower():
        report += "\n\n**Flag**: Confidence below threshold — requires human review."
    return {"needs_human_review": needs_review, "report_text": report}
