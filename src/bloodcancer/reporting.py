from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
import math

from bloodcancer.rag.tfidf_rag import Evidence

def _format_prob_table(prob_dict: Dict[str, float]) -> str:
    lines = []
    for k, v in sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True):
        try:
            lines.append(f"- {k}: {float(v):.4f}")
        except Exception:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else "- n/a"

def render_report(
    template_path: str,
    *,
    image_filename: str,
    predicted_label: str,
    confidence: float,
    probs: dict,
    explain_method: str,
    heatmap_path: str,
    explain_summary: str,
    evidences: List[Evidence],
    confidence_threshold: float,
    use_llm: bool = False,
    llm_text: str = "",
    # two-stage optional
    predicted_subtype: Optional[str] = None,
    subtype_confidence: Optional[float] = None,
    subtype_probs: Optional[dict] = None,
    subtype_disagreement: Optional[float] = None,
    stage1_snapshot: Optional[dict] = None,
) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")

    # Evidence block
    if evidences:
        ev_lines = []
        for e in evidences[:5]:
            ev_lines.append(f"- {e.source} (score={getattr(e, 'score', 0.0):.3f}): {getattr(e, 'snippet', '')}")
        evidence_block = "\n".join(ev_lines)
    else:
        evidence_block = "- n/a"

    # Interpretation + limitations
    needs_review = confidence < confidence_threshold
    interpretation = (
        "Low model confidence or high uncertainty. This case should be reviewed by a qualified clinician."
        if needs_review else
        "Model output indicates the predicted class with moderate/high confidence. Use alongside clinical context."
    )
    limitations = (
        "This system is for research decision-support. It may fail on out-of-distribution stains/scanners, rare morphologies, "
        "or dataset bias. Always validate locally and keep a human-in-the-loop."
    )

    # Subtype fields
    subtype_probs = subtype_probs or {}
    subtype_prob_table = _format_prob_table({k: float(v) for k, v in subtype_probs.items()}) if subtype_probs else "- n/a"
    stage1_snapshot = stage1_snapshot or {}

    out = tpl
    # basic
    out = out.replace("{{image_filename}}", str(image_filename))
    out = out.replace("{{predicted_label}}", str(predicted_label))
    out = out.replace("{{confidence}}", f"{float(confidence):.3f}")
    out = out.replace("{{prob_table}}", _format_prob_table({k: float(v) for k, v in (probs or {}).items()}))
    out = out.replace("{{explain_method}}", str(explain_method))
    out = out.replace("{{heatmap_path}}", str(heatmap_path))
    out = out.replace("{{explain_summary}}", str(explain_summary))
    out = out.replace("{{evidence_block}}", str(evidence_block))
    out = out.replace("{{interpretation}}", str(interpretation))
    out = out.replace("{{limitations}}", str(limitations))

    # subtype
    out = out.replace("{{predicted_subtype}}", str(predicted_subtype or "n/a"))
    out = out.replace("{{subtype_confidence}}", f"{float(subtype_confidence):.3f}" if subtype_confidence is not None else "n/a")
    out = out.replace("{{subtype_prob_table}}", subtype_prob_table)
    out = out.replace("{{subtype_disagreement}}", f"{float(subtype_disagreement):.4f}" if subtype_disagreement is not None else "n/a")

    # stage1 snapshot
    out = out.replace("{{stage1_predicted_label}}", str(stage1_snapshot.get("predicted_label") or "n/a"))
    out = out.replace("{{stage1_confidence}}", str(stage1_snapshot.get("confidence") or "n/a"))
    out = out.replace("{{stage1_model}}", str(stage1_snapshot.get("selected_model_name") or "n/a"))
    out = out.replace("{{stage1_checkpoint}}", str(stage1_snapshot.get("selected_checkpoint") or "n/a"))

    # optional LLM addendum
    if use_llm and llm_text:
        out = out + "\n\n---\n\n## LLM Addendum (Optional)\n" + llm_text.strip() + "\n"

    # Always include disclaimer
    out += "\n\n> **Not a diagnosis.** This output is for research decision-support only and must be reviewed by a qualified clinician.\n"
    return out
