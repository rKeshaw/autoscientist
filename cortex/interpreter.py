from __future__ import annotations

import json
from typing import Dict, Optional

from ollama import Client

OLLAMA_MODEL = "llama3.1:8b"


class ReflectiveInterpreter:
    """LLM is used only to narrate already-detected structural events."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.llm = Client() if enabled else None

    def summarize_insight(self, payload: Dict) -> str:
        if not self.enabled:
            return payload.get("default", "Insight detected.")
        prompt = (
            "You are a reflective interpreter, not a planner. "
            "Given this structural event in a cognitive graph, write one concise interpretation.\n"
            f"Event JSON: {json.dumps(payload)}"
        )
        try:
            response = self.llm.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"].strip()
        except Exception:
            return payload.get("default", "Insight detected.")

    def name_abstraction(self, members: list[str]) -> str:
        if not self.enabled:
            return f"Abstraction over {len(members)} ideas"
        prompt = (
            "Name this abstraction in <= 10 words.\n"
            f"Members: {members}"
        )
        try:
            response = self.llm.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"].strip()
        except Exception:
            return f"Abstraction over {len(members)} ideas"
