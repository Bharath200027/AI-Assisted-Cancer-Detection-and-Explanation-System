from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import os
import json

from bloodcancer.llm import call_llm
from bloodcancer.agents.tools import build_tool_registry, tool_trace_event

@dataclass
class SupervisorPolicy:
    name: str = "default"
    use_llm_planner: bool = False
    allow_tools: List[str] | None = None

DEFAULT_PLAN = [
    "routing",
    "triage",
    "preprocess_qc",
    "inference",
    "subtype_inference",
    "active_learning",
    "explain",
    "rag",
    "report",
    "safety_gate",
]

class SupervisorAgent:
    """Supervisor agent that orchestrates the pipeline using tools.

    Optional LLM planning (guardrailed):
    - If enabled, the LLM can propose *which tools to run* and in what order.
    - It must never diagnose, interpret medically, or prescribe treatment.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.registry = build_tool_registry(cfg)
        self.policy = SupervisorPolicy(
            name="default",
            use_llm_planner=bool(os.getenv("SUPERVISOR_USE_LLM", "0").lower() in {"1","true","yes"}),
            allow_tools=list(self.registry.keys()),
        )

    def propose_plan(self, state: Dict[str, Any]) -> List[str]:
        plan = list(DEFAULT_PLAN)

        if not self.cfg.rag_enabled and "rag" in plan:
            plan.remove("rag")

        if os.getenv("DISABLE_EXPLAIN", "0").lower() in {"1","true","yes"} and "explain" in plan:
            plan.remove("explain")

        if self.policy.use_llm_planner and self.cfg.report_use_llm:
            allow = ", ".join(sorted(self.policy.allow_tools or []))
            prompt = (
                "You are a workflow planner for a medical imaging decision-support pipeline.\n"
                "Task: Propose an ordered list of tools to run for one image.\n"
                "Constraints:\n"
                "- Do NOT diagnose or interpret medically.\n"
                f"- Only choose from this allowlist: {allow}\n"
                "- Always include: routing, triage, preprocess_qc, inference, report, safety_gate\n"
                "- Include subtype_inference if policy mode is two_stage (it can no-op).\n"
                "- Include active_learning unless probs/confidence unavailable.\n"
                'Return ONLY JSON: {"plan": ["tool1", "tool2", ...]}\n'
            )
            try:
                txt = call_llm(prompt)
                obj = json.loads(txt)
                cand = obj.get("plan", [])
                cand = [t for t in cand if t in (self.policy.allow_tools or [])]

                must = ["routing","triage","preprocess_qc","inference","report","safety_gate"]
                for m in must:
                    if m not in cand:
                        cand.append(m)

                if "subtype_inference" not in cand and "subtype_inference" in (self.policy.allow_tools or []):
                    if "inference" in cand:
                        cand.insert(cand.index("inference")+1, "subtype_inference")
                    else:
                        cand.append("subtype_inference")

                plan = cand
            except Exception:
                pass

        return plan

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        trace: List[dict] = state.get("tool_trace", []) or []
        plan = self.propose_plan(state)
        state["plan"] = plan

        for tool_name in plan:
            tool = self.registry.get(tool_name)
            if not tool:
                trace.append(tool_trace_event(tool_name, "skipped", {"reason": "unknown_tool"}))
                continue

            if state.get("errors") and tool_name not in {"report","safety_gate"}:
                trace.append(tool_trace_event(tool_name, "skipped", {"reason": "errors_present"}))
                continue

            trace.append(tool_trace_event(tool_name, "start"))
            try:
                updates = tool.run(state) or {}
                state.update(updates)

                if tool_name == "inference":
                    state["stage1_snapshot"] = {
                        "predicted_label": state.get("predicted_label"),
                        "confidence": state.get("confidence"),
                        "probs": state.get("probs"),
                        "selected_model_name": state.get("selected_model_name"),
                        "selected_checkpoint": state.get("selected_checkpoint"),
                    }

                trace.append(tool_trace_event(tool_name, "done", {"updated_keys": list(updates.keys())}))
            except Exception as e:
                errs = state.get("errors", []) or []
                errs.append(f"{tool_name} failed: {e}")
                state["errors"] = errs
                trace.append(tool_trace_event(tool_name, "error", {"error": str(e)}))

        state["tool_trace"] = trace
        return state
