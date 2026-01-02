from __future__ import annotations
from typing import Dict, Any

from bloodcancer.agents.agents import (
    TriageAgent, PreprocessQCAgent, InferenceAgent, ExplainabilityAgent, EvidenceRAGAgent, ReportAgent, SafetyGateAgent
)

def triage_node(state: Dict[str, Any], cfg) -> Dict[str, Any]:
    return TriageAgent(cfg).run(state)

def preprocess_node(state: Dict[str, Any], cfg) -> Dict[str, Any]:
    return PreprocessQCAgent(cfg).run(state)

def inference_node(state: Dict[str, Any], cfg) -> Dict[str, Any]:
    return InferenceAgent(cfg).run(state)

def explain_node(state: Dict[str, Any], cfg) -> Dict[str, Any]:
    return ExplainabilityAgent(cfg).run(state)

def rag_node(state: Dict[str, Any], cfg) -> Dict[str, Any]:
    return EvidenceRAGAgent(cfg).run(state)

def report_node(state: Dict[str, Any], cfg) -> Dict[str, Any]:
    return ReportAgent(cfg).run(state)

def safety_node(state: Dict[str, Any], cfg) -> Dict[str, Any]:
    return SafetyGateAgent(cfg).run(state)
