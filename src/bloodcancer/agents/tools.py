from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import datetime as dt
from pathlib import Path
import os
import json

from bloodcancer.registry import best_checkpoint, best_models_per_class, best_checkpoint_for_class, per_class_score

@dataclass
class ToolSpec:
    name: str
    purpose: str
    inputs: List[str]
    outputs: List[str]
    guardrails: List[str]

class Tool:
    spec: ToolSpec
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

def utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def tool_trace_event(tool_name: str, status: str, details: dict | None = None):
    return {
        "ts": utc_iso(),
        "tool": tool_name,
        "status": status,
        "details": details or {},
    }

# ------------------------------
# Agent wrappers (agents as tools)
# ------------------------------
from bloodcancer.agents.agents import (
    TriageAgent, PreprocessQCAgent, InferenceAgent, ExplainabilityAgent, EvidenceRAGAgent, ReportAgent, SafetyGateAgent
)

class TriageTool(Tool):
    def __init__(self, cfg):
        self.agent = TriageAgent(cfg)
        self.spec = ToolSpec("triage", self.agent.spec.proposition, self.agent.spec.inputs, self.agent.spec.outputs, self.agent.spec.guardrails)
    def run(self, state): return self.agent.run(state)

class PreprocessQCTool(Tool):
    def __init__(self, cfg):
        self.agent = PreprocessQCAgent(cfg)
        self.spec = ToolSpec("preprocess_qc", self.agent.spec.proposition, self.agent.spec.inputs, self.agent.spec.outputs, self.agent.spec.guardrails)
    def run(self, state): return self.agent.run(state)

class ExplainabilityTool(Tool):
    def __init__(self, cfg):
        self.agent = ExplainabilityAgent(cfg)
        self.spec = ToolSpec("explain", self.agent.spec.proposition, self.agent.spec.inputs, self.agent.spec.outputs, self.agent.spec.guardrails)
    def run(self, state): return self.agent.run(state)

class EvidenceRAGTool(Tool):
    def __init__(self, cfg):
        self.agent = EvidenceRAGAgent(cfg)
        self.spec = ToolSpec("rag", self.agent.spec.proposition, self.agent.spec.inputs, self.agent.spec.outputs, self.agent.spec.guardrails)
    def run(self, state): return self.agent.run(state)

class ReportTool(Tool):
    def __init__(self, cfg):
        self.agent = ReportAgent(cfg)
        self.spec = ToolSpec("report", self.agent.spec.proposition, self.agent.spec.inputs, self.agent.spec.outputs, self.agent.spec.guardrails)
    def run(self, state): return self.agent.run(state)

class SafetyGateTool(Tool):
    def __init__(self, cfg):
        self.agent = SafetyGateAgent(cfg)
        self.spec = ToolSpec("safety_gate", self.agent.spec.proposition, self.agent.spec.inputs, self.agent.spec.outputs, self.agent.spec.guardrails)
    def run(self, state): return self.agent.run(state)

# ------------------------------
# Policy routing & multi-model inference tools (v6)
# ------------------------------

def _load_models_cfg() -> dict:
    """Load configs/models.yaml (env override: MODELS_YAML)."""
    import yaml
    p = Path(os.getenv("MODELS_YAML", "configs/models.yaml"))
    if not p.exists():
        # Try resolving relative to repo root when cwd differs
        repo_root = Path(__file__).resolve().parents[3]
        alt = repo_root / "configs" / "models.yaml"
        if alt.exists():
            p = alt
        else:
            return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _candidate_from_registry_entry(entry: dict, *, policy: str, stage: str) -> dict:
    return {
        "id": entry.get("id"),
        "model_name": entry.get("model_name"),
        "checkpoint": entry.get("checkpoint"),
        "class_names": entry.get("class_names") or [],
        "_policy": policy,
        "_stage": stage,
        "_from_registry": True,
        "_metrics": entry.get("metrics") or {},
    }

def _select_best_for_id(policy: str, stage: str, cand: dict) -> dict:
    """Replace candidate checkpoint/model_name/class_names from registry best for that id (by accuracy)."""
    model_id = cand.get("id")
    best = best_checkpoint(policy, stage, model_id=model_id)
    if best and Path(best.get("checkpoint","")).exists():
        out = dict(cand)
        out.update({
            "checkpoint": best.get("checkpoint"),
            "model_name": best.get("model_name") or out.get("model_name"),
            "class_names": best.get("class_names") or out.get("class_names"),
            "_from_registry": True,
            "_metrics": best.get("metrics") or {},
        })
        return out
    return cand

def _stage_default_class_names(candidates: list[dict]) -> list[str]:
    # best effort: union preserving order
    seen = set()
    out = []
    for c in candidates:
        for cn in (c.get("class_names") or []):
            if cn not in seen:
                seen.add(cn)
                out.append(cn)
    return out

def _pick_stage_candidates(
    stage_obj: dict,
    *,
    policy_name: str,
    registry_stage: str,
) -> dict:
    """Return a normalized stage plan for inference."""
    ensemble = bool(stage_obj.get("ensemble", False))
    strategy = str(stage_obj.get("selection_strategy", "average")).lower()
    candidates_cfg = stage_obj.get("candidates", []) or []

    # Build candidate dicts and prefer best-per-id checkpoints from registry
    candidates: list[dict] = []
    for c in candidates_cfg:
        cand = dict(c)
        cand["_policy"] = policy_name
        cand["_stage"] = registry_stage
        cand = _select_best_for_id(policy_name, registry_stage, cand)
        candidates.append(cand)

    class_names = stage_obj.get("global_class_names") or _stage_default_class_names(candidates) or []
    class_to_model: dict[str, str] = {}

    # If no checkpoints exist on disk, keep candidates to allow a clear downstream error message
    existing = [c for c in candidates if Path(c.get("checkpoint","")).exists()]
    use = existing if existing else candidates

    selected = use
    if ensemble and strategy == "best_per_class" and class_names and use:
        # Select best checkpoint per class from registry among provided candidate IDs
        selected_map, selected_candidates = best_models_per_class(
            policy_name, registry_stage, class_names, use, metric="f1"
        )
        # Replace selected candidates with exact registry entries (ensures per-class selection uses correct checkpoint)
        # Build id->entry
        id_to_entry = {e.get("id"): e for e in selected_map.values()}
        uniq = []
        seen_ids = set()
        for c in selected_candidates:
            mid = c.get("id")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            # Prefer entry checkpoint if available for any class mapped to this id
            entry = id_to_entry.get(mid)
            if entry:
                uniq.append(_candidate_from_registry_entry(entry, policy=policy_name, stage=registry_stage))
            else:
                uniq.append(c)
        selected = uniq or selected_candidates
        class_to_model = {cn: e.get("id") for cn, e in selected_map.items() if e and e.get("id")}
    else:
        # default mapping: all classes use first model
        if use:
            class_to_model = {cn: use[0].get("id") for cn in class_names}

    return {
        "ensemble": ensemble,
        "selection_strategy": strategy,
        "registry_stage": registry_stage,
        "class_names": class_names,
        "candidates": use,
        "selected_candidates": selected,
        "class_to_model": class_to_model,
    }

class RoutingTool(Tool):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = ToolSpec(
            name="routing",
            purpose="Select pipeline policy and stage model plans (supports two-stage screening→subtype, hierarchical ensembles, and best-per-class selection).",
            inputs=["image_path"],
            outputs=[
                "policy","mode",
                "stage1_plan","stage2_plan",
                "selected_model_name","selected_checkpoint","class_names",
            ],
            guardrails=[
                "Never infer cancer type during routing; only select a pipeline and checkpoints.",
                "Prefer checkpoints that exist; otherwise downstream must raise a clear error."
            ],
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cfg = _load_models_cfg()
        policies = (cfg.get("policies") or {})
        policy_name = state.get("policy") or cfg.get("default_policy") or "binary"

        pol = policies.get(policy_name)
        if not pol:
            # fall back to env-based single model
            model_name = os.getenv("MODEL_NAME", "tf_efficientnetv2_s")
            ckpt = os.getenv("MODEL_CHECKPOINT", "artifacts/models/best.pt")
            class_names = self.cfg.class_names
            os.environ["MODEL_NAME"] = model_name
            os.environ["MODEL_CHECKPOINT"] = ckpt
            return {
                "policy": policy_name,
                "mode": "single",
                "stage1_plan": {"ensemble": False, "selection_strategy": "average", "registry_stage": "stage1", "class_names": class_names, "candidates": [{"id":"env_model","model_name":model_name,"checkpoint":ckpt,"class_names":class_names}], "selected_candidates": [{"id":"env_model","model_name":model_name,"checkpoint":ckpt,"class_names":class_names}], "class_to_model": {cn:"env_model" for cn in class_names}},
                "stage2_plan": None,
                "selected_model_name": model_name,
                "selected_checkpoint": ckpt,
                "class_names": class_names,
            }

        mode = str(pol.get("mode", "single")).lower()

        # Stage-1
        stage1_obj = pol.get("stage1") or {}
        stage1_plan = _pick_stage_candidates(stage1_obj, policy_name=policy_name, registry_stage="stage1")

        # Stage-2
        stage2_plan: dict | None = None
        if mode == "two_stage":
            st2 = pol.get("stage2") or {}
            st2_mode = str(st2.get("mode", "flat")).lower()
            if st2_mode == "flat":
                reg_stage = st2.get("registry_stage") or "stage2"
                stage2_plan = _pick_stage_candidates(st2, policy_name=policy_name, registry_stage=reg_stage)
                stage2_plan["mode"] = "flat"
            else:
                # hierarchical: families + fallback
                global_names = st2.get("global_class_names") or ["all","aml","cll","cml"]
                families_cfg = st2.get("families") or {}
                family_plans = []
                for fam_name, fam_cfg in families_cfg.items():
                    reg_stage = fam_cfg.get("registry_stage") or f"stage2_{fam_name}"
                    plan = _pick_stage_candidates(fam_cfg, policy_name=policy_name, registry_stage=reg_stage)
                    plan["family"] = fam_name
                    plan["mode"] = "family"
                    family_plans.append(plan)

                fb_cfg = st2.get("fallback") or {}
                fb_plan = None
                if fb_cfg:
                    reg_stage = fb_cfg.get("registry_stage") or "stage2"
                    fb_plan = _pick_stage_candidates(fb_cfg, policy_name=policy_name, registry_stage=reg_stage)
                    fb_plan["mode"] = "fallback"

                stage2_plan = {
                    "mode": "hierarchical",
                    "selection_strategy": str(st2.get("selection_strategy", "best_per_class")).lower(),
                    "global_class_names": global_names,
                    "families": family_plans,
                    "fallback": fb_plan,
                }

        # For backward compatibility with older nodes/agents, keep selected_* pointed to stage1 first model
        sel = (stage1_plan.get("selected_candidates") or stage1_plan.get("candidates") or [{}])[0]
        selected_model_name = sel.get("model_name", os.getenv("MODEL_NAME", "tf_efficientnetv2_s"))
        selected_checkpoint = sel.get("checkpoint", os.getenv("MODEL_CHECKPOINT", "artifacts/models/best.pt"))
        class_names = stage1_plan.get("class_names") or self.cfg.class_names

        os.environ["MODEL_NAME"] = str(selected_model_name)
        os.environ["MODEL_CHECKPOINT"] = str(selected_checkpoint)

        return {
            "policy": policy_name,
            "mode": mode,
            # new structured plans
            "stage1_plan": stage1_plan,
            "stage2_plan": stage2_plan,
            # backward-compatible flat keys
            "stage1_ensemble": bool(stage1_plan.get("ensemble")),
            "stage1_selection_strategy": stage1_plan.get("selection_strategy"),
            "stage1_class_to_model": stage1_plan.get("class_to_model"),
            "stage1_candidates": stage1_plan.get("selected_candidates") or stage1_plan.get("candidates") or [],
            "stage2_mode": (stage2_plan or {}).get("mode") if isinstance(stage2_plan, dict) else None,
            "stage2_candidates": (stage2_plan or {}).get("selected_candidates") if isinstance(stage2_plan, dict) and (stage2_plan.get("mode") == "flat") else [],
            "stage2_ensemble": bool((stage2_plan or {}).get("ensemble")) if isinstance(stage2_plan, dict) and (stage2_plan.get("mode") == "flat") else False,
            "stage2_selection_strategy": (stage2_plan or {}).get("selection_strategy") if isinstance(stage2_plan, dict) and (stage2_plan.get("mode") == "flat") else None,
            "stage2_class_to_model": (stage2_plan or {}).get("class_to_model") if isinstance(stage2_plan, dict) and (stage2_plan.get("mode") == "flat") else None,
            # selected model for legacy env overrides
            "selected_model_name": selected_model_name,
            "selected_checkpoint": selected_checkpoint,
            "class_names": class_names,
        }
def _run_single_inference(cfg, state: Dict[str, Any], model_name: str, checkpoint: str, class_names: list[str]) -> Dict[str, Any]:
    # InferenceAgent supports overrides in state keys
    local_state = dict(state)
    local_state.update({
        "selected_model_name": model_name,
        "selected_checkpoint": checkpoint,
        "class_names": class_names,
    })
    return InferenceAgent(cfg).run(local_state)

def _ensemble_combine(
    outputs: list[Dict[str, Any]],
    class_names: list[str],
    strategy: str = "average",
    class_to_model: dict[str, str] | None = None,
) -> tuple[dict, float, float]:
    """Combine ensemble outputs.

    strategies:
      - average: mean probabilities
      - best_per_class: choose per-class prob from selected model (class_to_model)
    """
    import numpy as np

    if not outputs:
        return {}, 0.0, 0.0

    # Build model->vector matrix
    model_ids = []
    mats = []
    for out in outputs:
        probs = out.get("probs") or {}
        vec = [float(probs.get(cn, 0.0)) for cn in class_names]
        mats.append(vec)
        model_ids.append(str(out.get("_model_id") or out.get("model_id") or ""))

    M = np.asarray(mats, dtype=np.float32) if mats else np.zeros((1, len(class_names)), dtype=np.float32)
    # disagreement: std of max probs across models
    maxes = M.max(axis=1) if M.size else np.array([0.0])
    disagree = float(maxes.std())

    strategy = (strategy or "average").lower().strip()
    if strategy == "best_per_class" and class_to_model:
        # choose class-wise from mapped model
        probs_out = {}
        for j, cn in enumerate(class_names):
            mid = str(class_to_model.get(cn) or "")
            if mid and mid in model_ids:
                i = model_ids.index(mid)
                probs_out[cn] = float(M[i, j])
            else:
                probs_out[cn] = float(M[:, j].mean())
        # normalize
        s = sum(probs_out.values())
        if s > 0:
            probs_out = {k: float(v / s) for k, v in probs_out.items()}
        top = max(probs_out, key=probs_out.get)
        conf = float(probs_out[top])
        return probs_out, conf, disagree

    # average
    avg = M.mean(axis=0)
    top_idx = int(avg.argmax()) if avg.size else 0
    conf = float(avg[top_idx]) if avg.size else 0.0
    probs_avg = {class_names[i]: float(avg[i]) for i in range(len(class_names))}
    # normalize
    s = sum(probs_avg.values())
    if s > 0:
        probs_avg = {k: float(v / s) for k, v in probs_avg.items()}
        conf = float(max(probs_avg.values()))
    return probs_avg, conf, disagree

class InferenceTool(Tool):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = ToolSpec(
            name="inference",
            purpose="Run stage-1 inference (screening or multiclass). Supports ensembles.",
            inputs=["image_path","stage1_candidates","stage1_ensemble"],
            outputs=["probs","predicted_label","confidence","disagreement"],
            guardrails=[
                "If checkpoints missing, record error and stop (no guessing).",
                "Return probabilities aligned to the selected class_names."
            ],
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        cands = state.get("stage1_candidates") or []
        if not cands:
            return InferenceAgent(self.cfg).run(state)

        # Use class_names from first candidate (must match across ensemble)
        class_names = cands[0].get("class_names") or state.get("class_names") or self.cfg.class_names
        outs = []
        run_list = cands if state.get("stage1_ensemble") else cands[:1]
        for c in run_list:
            out = _run_single_inference(self.cfg, state, c.get("model_name"), c.get("checkpoint"), class_names)
            if out.get("errors"):
                return out
            out["_model_id"] = c.get("id")
            outs.append(out)

        if len(outs) == 1:
            out = outs[0]
            out["disagreement"] = 0.0
            # persist selected model for later explainability
            out.update({"selected_model_name": cands[0].get("model_name"), "selected_checkpoint": cands[0].get("checkpoint"), "class_names": class_names,
                        "stage1_snapshot": {"predicted_label": out.get("predicted_label"), "confidence": out.get("confidence"), "selected_model_name": cands[0].get("model_name"), "selected_checkpoint": cands[0].get("checkpoint"), "disagreement": out.get("disagreement", 0.0)}})
            return out

        probs, conf, disagree = _ensemble_combine(outs, class_names, strategy=state.get('stage1_selection_strategy','average'), class_to_model=state.get('stage1_class_to_model') or {})
        pred_label = max(probs, key=probs.get)
        return {
            "probs": probs,
            "predicted_label": pred_label,
            "confidence": conf,
            "disagreement": disagree,
            "selected_model_name": "ensemble",
            "selected_checkpoint": ",".join([c.get("checkpoint","") for c in cands]),
            "class_names": class_names,
            "stage1_snapshot": {
                "predicted_label": pred_label,
                "confidence": conf,
                "selected_model_name": "ensemble",
                "selected_checkpoint": ",".join([c.get("checkpoint","") for c in cands]),
                "disagreement": disagree,
            },
        }

class SubtypeInferenceTool(Tool):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = ToolSpec(
            name="subtype_inference",
            purpose="Run stage-2 subtype inference (ALL/AML/CLL/CML) if policy=two_stage and stage1 predicts leukemia. Supports ensembles.",
            inputs=["policy","mode","predicted_label","stage2_candidates","stage2_ensemble"],
            outputs=["predicted_subtype","subtype_confidence","subtype_probs","subtype_disagreement"],
            guardrails=[
                "Only runs when stage1 predicted leukemia.",
                "Does not run without subtype model candidates."
            ],
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("errors"):
            return {}
        if state.get("mode") != "two_stage":
            return {}
        if (state.get("predicted_label") or "").lower() != "leukemia":
            return {}

        stage2_plan = state.get("stage2_plan")

        def _run_plan(plan: dict) -> tuple[dict, float, float, list[str], list[str]]:
            """Return (probs, conf, disagree, model_ids, checkpoints)."""
            cands = plan.get("selected_candidates") or plan.get("candidates") or []
            if not cands:
                return {}, 0.0, 0.0, [], []
            class_names = plan.get("class_names") or (cands[0].get("class_names") if cands else []) or ["all","aml","cll","cml"]
            strategy = str(plan.get("selection_strategy") or "average").lower()
            class_to_model = plan.get("class_to_model") or {}
            ensemble = bool(plan.get("ensemble", False))

            outs = []
            run_list = cands if ensemble else cands[:1]
            mids, ckpts = [], []
            for c in run_list:
                out = _run_single_inference(self.cfg, state, c.get("model_name"), c.get("checkpoint"), class_names)
                if out.get("errors"):
                    raise RuntimeError("; ".join(out.get("errors")))
                out["_model_id"] = c.get("id")
                outs.append(out)
                mids.append(str(c.get("id") or ""))
                ckpts.append(str(c.get("checkpoint") or ""))

            if len(outs) == 1:
                probs = outs[0].get("probs") or {}
                pred = outs[0].get("predicted_label") or (max(probs, key=probs.get) if probs else None)
                conf = float(outs[0].get("confidence") or (probs.get(pred) if pred else 0.0) or 0.0)
                return probs, conf, 0.0, mids, ckpts

            probs, conf, disagree = _ensemble_combine(outs, class_names, strategy=strategy, class_to_model=class_to_model)
            return probs, conf, disagree, mids, ckpts

        # Hierarchical path
        if isinstance(stage2_plan, dict) and stage2_plan.get("mode") == "hierarchical":
            global_names = stage2_plan.get("global_class_names") or ["all","aml","cll","cml"]
            family_plans = stage2_plan.get("families") or []
            fallback = stage2_plan.get("fallback")

            # Run families that have at least one existing checkpoint
            ran_any = False
            global_probs = {cn: 0.0 for cn in global_names}
            disagreements = []
            used_model_ids = []
            used_checkpoints = []
            group_summaries = []

            for plan in family_plans:
                cands = plan.get("selected_candidates") or plan.get("candidates") or []
                if not any(Path(c.get("checkpoint","")).exists() for c in cands):
                    continue
                try:
                    probs, conf, dis, mids, ckpts = _run_plan(plan)
                except Exception as e:
                    # if a family fails, skip it and rely on fallback if available
                    group_summaries.append({"family": plan.get("family"), "error": str(e)})
                    continue

                if probs:
                    ran_any = True
                    for cn, pv in probs.items():
                        if cn in global_probs:
                            global_probs[cn] = float(pv)
                    disagreements.append(float(dis))
                    used_model_ids.extend(mids)
                    used_checkpoints.extend(ckpts)
                    group_summaries.append({"family": plan.get("family"), "probs": probs, "disagreement": float(dis)})

            if not ran_any:
                # Use fallback (flat) if provided, else fall back to legacy stage2_candidates
                if isinstance(fallback, dict):
                    probs, conf, dis, mids, ckpts = _run_plan(fallback)
                    global_probs = {cn: float(probs.get(cn, 0.0)) for cn in global_names}
                    disagreements = [float(dis)]
                    used_model_ids = mids
                    used_checkpoints = ckpts
                    group_summaries = [{"family": "fallback", "probs": probs, "disagreement": float(dis)}]
                else:
                    # legacy keys
                    cands = state.get("stage2_candidates") or []
                    if not cands:
                        return {}
                    plan = {
                        "ensemble": bool(state.get("stage2_ensemble")),
                        "selection_strategy": state.get("stage2_selection_strategy") or "average",
                        "class_to_model": state.get("stage2_class_to_model") or {},
                        "class_names": cands[0].get("class_names") or ["all","aml","cll","cml"],
                        "selected_candidates": cands,
                    }
                    probs, conf, dis, mids, ckpts = _run_plan(plan)
                    global_probs = {cn: float(probs.get(cn, 0.0)) for cn in global_names}
                    disagreements = [float(dis)]
                    used_model_ids = mids
                    used_checkpoints = ckpts
                    group_summaries = [{"family": "legacy_flat", "probs": probs, "disagreement": float(dis)}]

            # normalize
            s = sum(global_probs.values())
            if s > 0:
                global_probs = {k: float(v / s) for k, v in global_probs.items()}

            pred = max(global_probs, key=global_probs.get) if global_probs else None
            conf = float(global_probs.get(pred, 0.0)) if pred else 0.0
            dis = float(sum(disagreements) / len(disagreements)) if disagreements else 0.0

            return {
                "predicted_subtype": pred,
                "subtype_confidence": conf,
                "subtype_probs": global_probs,
                "subtype_disagreement": dis,
                "subtype_mode": "hierarchical",
                "subtype_groups": group_summaries,
                "subtype_models_used": used_model_ids,
                "subtype_checkpoints_used": used_checkpoints,
            }

        # Flat path (legacy)
        cands = state.get("stage2_candidates") or []
        if not cands:
            # if stage2_plan exists but isn't hierarchical, treat as flat plan
            if isinstance(stage2_plan, dict) and stage2_plan.get("mode") == "flat":
                cands = stage2_plan.get("selected_candidates") or stage2_plan.get("candidates") or []
            if not cands:
                return {}

        class_names = cands[0].get("class_names") or ["all","aml","cll","cml"]
        outs = []
        run_list = cands if state.get("stage2_ensemble") else cands[:1]
        for c in run_list:
            out = _run_single_inference(self.cfg, state, c.get("model_name"), c.get("checkpoint"), class_names)
            if out.get("errors"):
                return {"errors": state.get("errors", []) + out.get("errors", [])}
            out["_model_id"] = c.get("id")
            outs.append(out)

        if len(outs) == 1:
            probs = outs[0].get("probs") or {}
            pred = outs[0].get("predicted_label") or (max(probs, key=probs.get) if probs else None)
            conf = float(outs[0].get("confidence") or (probs.get(pred) if pred else 0.0) or 0.0)
            return {
                "predicted_subtype": pred,
                "subtype_confidence": conf,
                "subtype_probs": probs,
                "subtype_disagreement": 0.0,
                "subtype_mode": "flat",
            }

        probs, conf, disagree = _ensemble_combine(
            outs,
            class_names,
            strategy=state.get("stage2_selection_strategy","average"),
            class_to_model=state.get("stage2_class_to_model") or {},
        )
        pred = max(probs, key=probs.get) if probs else None
        return {
            "predicted_subtype": pred,
            "subtype_confidence": conf,
            "subtype_probs": probs,
            "subtype_disagreement": disagree,
            "subtype_mode": "flat",
        }

class ActiveLearningTool(Tool):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = ToolSpec(
            name="active_learning",
            purpose="Compute uncertainty score and decide whether this case should be prioritized for human labeling.",
            inputs=["probs", "confidence", "disagreement"],
            outputs=["active_learning_score", "suggested_for_labeling"],
            guardrails=[
                "Do not change prediction; only compute prioritization metadata.",
                "Use only model probabilities; no external patient data."
            ],
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        probs = state.get("probs") or {}
        if not probs:
            return {}
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top1 = float(items[0][1])
        top2 = float(items[1][1]) if len(items) > 1 else 0.0
        uncertainty = 1.0 - top1
        margin = top1 - top2
        disagree = float(state.get("disagreement") or 0.0)
        score = float(uncertainty + (0.5 * (1.0 - margin)) + (0.5 * disagree))
        suggested = bool(top1 < max(0.85, self.cfg.confidence_threshold + 0.05) or margin < 0.15 or disagree > 0.08)
        return {"active_learning_score": score, "suggested_for_labeling": suggested}

def build_tool_registry(cfg) -> Dict[str, Tool]:
    return {
        "routing": RoutingTool(cfg),
        "triage": TriageTool(cfg),
        "preprocess_qc": PreprocessQCTool(cfg),
        "inference": InferenceTool(cfg),
        "subtype_inference": SubtypeInferenceTool(cfg),
        "active_learning": ActiveLearningTool(cfg),
        "explain": ExplainabilityTool(cfg),
        "rag": EvidenceRAGTool(cfg),
        "report": ReportTool(cfg),
        "safety_gate": SafetyGateTool(cfg),
    }

def list_tool_specs(cfg) -> List[dict]:
    reg = build_tool_registry(cfg)
    out = []
    for t in reg.values():
        out.append({
            "name": t.spec.name,
            "purpose": t.spec.purpose,
            "inputs": t.spec.inputs,
            "outputs": t.spec.outputs,
            "guardrails": t.spec.guardrails,
        })
    return sorted(out, key=lambda x: x["name"])
