# Deprecated: node logic moved to src/bloodcancer/agents/agents.py and src/bloodcancer/agents/nodes.py
from __future__ import annotations
from bloodcancer.rag.tfidf_rag import TfidfRAG

def rag_node(state: dict, cfg) -> dict:
    if state.get("errors") or not cfg.rag_enabled:
        return {"evidences": []}
    rag = TfidfRAG(cfg.kb_dir, top_k=cfg.rag_top_k)
    query = f"blood smear leukemia classification limitations explainability confidence {state.get('predicted_label','')}"
    ev = rag.search(query)
    # Convert dataclasses to plain dicts for LangGraph state
    out = [{"source": e.source, "score": e.score, "snippet": e.snippet} for e in ev]
    return {"evidences": out}
