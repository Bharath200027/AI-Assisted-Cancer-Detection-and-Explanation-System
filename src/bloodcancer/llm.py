from __future__ import annotations
import os
from typing import Optional

def call_llm(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    if provider in {"none", ""}:
        return ""

    if provider == "openai":
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package not installed. Install with: pip install openai") from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        client = OpenAI(api_key=api_key)
        # Keep messages minimal & grounded
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system", "content":"You are a medical AI assistant that writes cautious, non-diagnostic decision-support summaries. Never prescribe treatment. Always mention uncertainty and need for clinician review."},
                {"role":"user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    raise RuntimeError(f"Unsupported LLM_PROVIDER={provider}")
