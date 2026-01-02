from __future__ import annotations
from typing import TypedDict, Any

from bloodcancer.config import AppConfig, load_yaml
from bloodcancer.agents.supervisor import SupervisorAgent

try:
    from langgraph.graph import StateGraph, END
except Exception as e:
    StateGraph = None
    END = None

class CaseState(TypedDict, total=False):
    image_path: str
    modality: str
    image_filename: str
    probs: dict
    predicted_label: str
    confidence: float
    heatmap_path: str
    explain_summary: str
    evidences: list
    report_text: str
    needs_human_review: bool
    active_learning_score: float
    suggested_for_labeling: bool
    plan: list
    tool_trace: list
    errors: list

def create_graph(config_path: str = "configs/app.yaml"):
    cfg = AppConfig(load_yaml(config_path))

    if StateGraph is None:
        raise RuntimeError("langgraph is not installed. Install with: pip install langgraph")

    g = StateGraph(CaseState)

    def supervisor_exec(state: dict) -> dict:
        sup = SupervisorAgent(cfg)
        return sup.execute(state)

    g.add_node("supervisor", supervisor_exec)
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", END)
    return g.compile()
