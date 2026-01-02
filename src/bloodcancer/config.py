from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import yaml

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def get_device(device_cfg: str) -> str:
    import torch
    if device_cfg == "cpu":
        return "cpu"
    if device_cfg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"

def getenv_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1","true","yes","y","on"}

@dataclass
class AppConfig:
    raw: dict

    @property
    def class_names(self) -> list[str]:
        return list(self.raw["app"]["class_names"])

    @property
    def confidence_threshold(self) -> float:
        return float(self.raw["inference"]["confidence_threshold"])

    @property
    def img_size(self) -> int:
        return int(self.raw["inference"]["img_size"])

    @property
    def device(self) -> str:
        return get_device(str(self.raw["app"]["device"]))

    @property
    def rag_enabled(self) -> bool:
        return bool(self.raw.get("rag", {}).get("enabled", True))

    @property
    def kb_dir(self) -> str:
        return str(self.raw.get("rag", {}).get("knowledge_base_dir", "knowledge_base"))

    @property
    def rag_top_k(self) -> int:
        return int(self.raw.get("rag", {}).get("top_k", 4))

    @property
    def report_template(self) -> str:
        return str(self.raw.get("report", {}).get("template_path", "configs/prompts/report_template.md"))

    @property
    def report_use_llm(self) -> bool:
        # allow env override
        return getenv_bool("USE_LLM", bool(self.raw.get("report", {}).get("use_llm", False)))

    @property
    def llm_provider(self) -> str:
        return os.getenv("LLM_PROVIDER", str(self.raw.get("report", {}).get("llm_provider", "none")))

    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", str(self.raw.get("report", {}).get("openai_model", "gpt-4o-mini")))
