from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import json

import torch
from PIL import Image
from bloodcancer.vision.preprocess import preprocess_pil, batchify
from bloodcancer.models.modeling import create_classifier, load_checkpoint
from bloodcancer.explain.gradcam import gradcam_explain
from bloodcancer.rag.tfidf_rag import TfidfRAG
from bloodcancer.reporting import render_report
from bloodcancer.llm import call_llm

# -------------------------
# Base Agent
# -------------------------

@dataclass
class AgentSpec:
    name: str
    proposition: str
    inputs: List[str]
    outputs: List[str]
    guardrails: List[str]

class BaseAgent:
    """A simple, explicit 'agent' interface for agentic workflows.
    Each agent:
      - has a Proposition (what it tries to achieve),
      - expects certain inputs in the shared state,
      - returns a dict of updates to the state,
      - must follow guardrails.
    """
    spec: AgentSpec

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

# -------------------------
# Agents
# -------------------------

class TriageAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = AgentSpec(
            name="TriageAgent",
            proposition="Validate the request and identify the modality + basic metadata for the case.",
            inputs=["image_path"],
            outputs=["modality", "image_filename"],
            guardrails=["If image_path is missing, add error and stop downstream work."]
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        image_path = state.get("image_path")
        if not image_path:
            return {"errors": state.get("errors", []) + ["image_path missing"]}
        p = Path(image_path)
        return {
            "modality": self.cfg.raw["app"].get("modality", "blood_smear"),
            "image_filename": p.name,
        }

class PreprocessQCAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = AgentSpec(
            name="PreprocessQCAgent",
            proposition="Run deterministic QC checks to catch invalid/low-quality inputs before inference.",
            inputs=["image_path"],
            outputs=["errors"],
            guardrails=[
                "Do not modify image pixels silently; only validate and record issues.",
                "If QC fails, append to errors and downstream agents should no-op."
            ]
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        image_path = state["image_path"]
        try:
            img = Image.open(image_path).convert("RGB")
            arr = __import__("numpy").array(img)
            h, w = arr.shape[:2]
            errs = []
            if h < 64 or w < 64:
                errs.append(f"Image too small: {w}x{h}")
            if float(arr.std()) < 2.0:
                errs.append("Image appears nearly uniform (low contrast).")
            if errs:
                return {"errors": state.get("errors", []) + errs}
            return {}
        except Exception as e:
            return {"errors": state.get("errors", []) + [f"Failed to load image: {e}"]}

class InferenceAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = AgentSpec(
            name="InferenceAgent",
            proposition="Run the deep learning model to predict blood cancer type probabilities.",
            inputs=["image_path"],
            outputs=["probs", "predicted_label", "confidence"],
            guardrails=[
                "If checkpoint missing, record error (do not guess).",
                "Return probabilities aligned to cfg.class_names."
            ]
        )

    def _preprocess(self, img: Image.Image, img_size: int):
        t = transforms.Compose([
        ])
        return t(img).unsqueeze(0)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        ckpt = state.get("selected_checkpoint") or os.getenv("MODEL_CHECKPOINT", "artifacts/models/best.pt")
        model_name = state.get("selected_model_name") or os.getenv("MODEL_NAME", "tf_efficientnetv2_s")
        class_names = state.get("class_names") or self.cfg.class_names
        device = self.cfg.device
        img_size = self.cfg.img_size

        if not Path(ckpt).exists():
            return {"errors": state.get("errors", []) + [f"Checkpoint not found: {ckpt}. Train first or set MODEL_CHECKPOINT."]}

        model = create_classifier(model_name, num_classes=len(class_names), pretrained=False)
        load_checkpoint(model, ckpt, map_location=device)
        model.to(device).eval()

        img = Image.open(state["image_path"]).convert("RGB")
        xb = self._preprocess(img, img_size).to(device)

        with torch.no_grad():
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()

        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        pred_label = class_names[pred_idx]
        conf = float(probs[pred_idx])

        return {
            "probs": {class_names[i]: float(p) for i, p in enumerate(probs)},
            "predicted_label": pred_label,
            "confidence": conf,
        }

class ExplainabilityAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = AgentSpec(
            name="ExplainabilityAgent",
            proposition="Generate a visual explanation (Grad-CAM) to highlight regions correlated with the prediction.",
            inputs=["image_path", "predicted_label"],
            outputs=["heatmap_path", "explain_summary"],
            guardrails=[
                "Explanations must be labeled as correlational (not causal).",
                "If errors exist, do nothing."
            ]
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        out_dir = Path(os.getenv("OUT_DIR", "artifacts"))
        out_path = out_dir / "explanations" / f"gradcam_{Path(state['image_path']).stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = state.get("selected_checkpoint") or os.getenv("MODEL_CHECKPOINT", "artifacts/models/best.pt")
        model_name = state.get("selected_model_name") or os.getenv("MODEL_NAME", "tf_efficientnetv2_s")
        device = self.cfg.device

        model = create_classifier(model_name, num_classes=len(self.cfg.class_names), pretrained=False)
        load_checkpoint(model, ckpt, map_location=device)
        model.to(device).eval()

        img = Image.open(state["image_path"]).convert("RGB")
        class_names = state.get("class_names") or self.cfg.class_names
        class_idx = class_names.index(state["predicted_label"])
        summary = gradcam_explain(model, img, class_idx, str(out_path), device=device, img_size=self.cfg.img_size)

        return {"heatmap_path": str(out_path), "explain_summary": summary}

class EvidenceRAGAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self._rag = TfidfRAG(self.cfg.kb_dir, top_k=self.cfg.rag_top_k) if self.cfg.rag_enabled else None
        self.spec = AgentSpec(
            name="EvidenceRAGAgent",
            proposition="Retrieve grounded evidence snippets about dataset/model limitations and safe interpretation.",
            inputs=["predicted_label"],
            outputs=["evidences"],
            guardrails=[
                "Only retrieve from local knowledge_base (grounding).",
                "If RAG disabled, return empty."
            ]
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors") or not self._rag:
            return {"evidences": []}
        q = f"blood smear leukemia classification limitations explainability confidence {state.get('predicted_label','')}"
        ev = self._rag.search(q)
        out = [{"source": e.source, "score": e.score, "snippet": e.snippet} for e in ev]
        return {"evidences": out}

class ReportAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = AgentSpec(
            name="ReportAgent",
            proposition="Generate a cautious, structured decision-support report grounded in model outputs + retrieved evidence.",
            inputs=["image_filename","predicted_label","confidence","probs","heatmap_path","explain_summary","evidences"],
            outputs=["report_text"],
            guardrails=[
                "Must include 'Not a diagnosis' disclaimer.",
                "Never prescribe treatment.",
                "If confidence below threshold, clearly flag for human review."
            ]
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors"):
            report = "Errors encountered:\n" + "\n".join(f"- {e}" for e in state["errors"])
            report += "\n\n> **Not a diagnosis.** This output is for research/decision-support only and must be reviewed by a qualified clinician.\n"
            return {"report_text": report, "needs_human_review": True}

        # Convert evidence dicts to attribute objects expected by render_report
        evidences = []
        for e in state.get("evidences", []) or []:
            evidences.append(type("Evidence", (), e))

        llm_text = ""
        if self.cfg.report_use_llm:
            prompt = (
                "Draft a cautious decision-support interpretation for a blood smear image classification.\n"
                f"Predicted label: {state['predicted_label']}\n"
                f"Confidence: {state['confidence']:.3f}\n"
                f"Explainability summary: {state.get('explain_summary','')}\n"
                "Constraints: Not a diagnosis; never prescribe treatment; mention uncertainty; recommend clinician review.\n"
            )
            llm_text = call_llm(prompt)

        report = render_report(
            self.cfg.report_template,
            image_filename=state.get("image_filename", Path(state["image_path"]).name),
            predicted_label=state["predicted_label"],
            confidence=state["confidence"],
            probs=state["probs"],
            predicted_subtype=state.get("predicted_subtype"),
            subtype_confidence=state.get("subtype_confidence"),
            subtype_probs=state.get("subtype_probs"),
            subtype_disagreement=state.get("subtype_disagreement"),
            stage1_snapshot=state.get("stage1_snapshot"),
            explain_method="Grad-CAM",
            heatmap_path=state.get("heatmap_path", ""),
            explain_summary=state.get("explain_summary", ""),
            evidences=evidences,
            confidence_threshold=self.cfg.confidence_threshold,
            use_llm=self.cfg.report_use_llm,
            llm_text=llm_text,
        )
        return {"report_text": report}

class SafetyGateAgent(BaseAgent):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = AgentSpec(
            name="SafetyGateAgent",
            proposition="Apply safety checks: confidence gating, enforce disclaimers, and mark cases for human review.",
            inputs=["confidence","report_text"],
            outputs=["needs_human_review","report_text"],
            guardrails=[
                "If confidence < threshold => needs_human_review True.",
                "Disclaimers must be present."
            ]
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors"):
            return {"needs_human_review": True}
        needs_review = bool(state.get("confidence", 0.0) < self.cfg.confidence_threshold)
        report = state.get("report_text", "")
        if "Not a diagnosis" not in report:
            report += "\n\n> **Not a diagnosis.** This output is for research/decision-support only and must be reviewed by a qualified clinician.\n"
        if needs_review and "human review" not in report.lower():
            report += "\n\n**Flag**: Confidence below threshold — requires human review."
        return {"needs_human_review": needs_review, "report_text": report}

# -------------------------
# Registry helper (for UI / debugging)
# -------------------------

def list_agent_specs(cfg) -> List[dict]:
    agents = [
        TriageAgent(cfg),
        PreprocessQCAgent(cfg),
        InferenceAgent(cfg),
        ExplainabilityAgent(cfg),
        EvidenceRAGAgent(cfg),
        ReportAgent(cfg),
        SafetyGateAgent(cfg),
    ]
    return [{
        "name": a.spec.name,
        "proposition": a.spec.proposition,
        "inputs": a.spec.inputs,
        "outputs": a.spec.outputs,
        "guardrails": a.spec.guardrails,
    } for a in agents]
