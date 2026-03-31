"""
LLM Utilities — Robust JSON parsing, multi-model support, and shared LLM interface.

All modules should use these utilities instead of raw ollama.Client calls
and bare json.loads() to ensure consistent behavior and error handling.
"""

import re
import json
import time
from ollama import Client

# ── Singleton client ──────────────────────────────────────────────────────────

_client = None

def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


# ── Robust JSON parsing ──────────────────────────────────────────────────────

def parse_llm_json(raw: str):
    """
    Extract JSON from LLM output, handling common failure modes:
    - Markdown code fences (```json ... ```)
    - Preamble text before JSON
    - Trailing text after JSON
    - Single-quoted strings (common with smaller models)

    Returns parsed object or None if truly unparseable.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object or array within the text
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start < 0:
            continue

        # Find matching closing bracket, handling nesting
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        # Try fixing single quotes
                        try:
                            fixed = candidate.replace("'", '"')
                            return json.loads(fixed)
                        except (json.JSONDecodeError, ValueError):
                            break

    return None


def require_json(raw: str, default=None):
    """Parse LLM JSON output, returning default if unparseable."""
    result = parse_llm_json(raw)
    return result if result is not None else default


# ── Multi-model LLM calls ────────────────────────────────────────────────────

def llm_call(prompt: str, temperature: float = 0.7,
             model: str = None, system: str = None,
             role: str = "creative") -> str:
    """
    Unified LLM call with model selection based on task role.

    Roles:
        creative  — dreaming, synthesis, analogies (higher temp, creative model)
        precise   — JSON extraction, factual questions (low temp, precise model)
        code      — code generation for sandbox
        reasoning — deliberate thinking, chain-of-thought
    """
    from config import MODELS

    if model is None:
        model = getattr(MODELS, role.upper(), MODELS.CREATIVE)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = _get_client()
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature}
    )
    return response['message']['content'].strip()


def llm_json(prompt: str, temperature: float = 0.1,
             model: str = None, default=None,
             system: str = None) -> any:
    """
    LLM call that expects JSON output. Uses precise model by default.

    Always uses the JSON system message for better compliance.
    Returns parsed JSON or default if unparseable.
    """
    json_system = (
        "You are a structured data extractor. You respond ONLY with valid JSON. "
        "No preamble, no explanation, no markdown code blocks, no trailing text. "
        "Just the raw JSON object or array."
    )
    if system:
        json_system = system + "\n\n" + json_system

    raw = llm_call(
        prompt,
        temperature=temperature,
        model=model,
        system=json_system,
        role="precise"
    )
    return require_json(raw, default=default)


def llm_chat(messages: list[dict], temperature: float = 0.7,
             model: str = None, role: str = "creative") -> str:
    """
    Multi-turn LLM call for conversation-style interactions.
    """
    from config import MODELS

    if model is None:
        model = getattr(MODELS, role.upper(), MODELS.CREATIVE)

    client = _get_client()
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature}
    )
    return response['message']['content'].strip()
