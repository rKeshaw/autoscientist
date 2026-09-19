import re
import json
from typing import Optional

_client = None


def _get_client():
    global _client
    if _client is None:
        import ollama
        _client = ollama.Client()
    return _client


# ── JSON parsing helpers ─────────────────────────────────────────────────────

def parse_llm_json(raw: str):
    """
    Extract JSON from LLM output, handling common failure modes:
    - Markdown code fences (```json ... ```) with preamble or trailing text
    - Braces or brackets inside string literals
    - Trailing commas before closing braces/brackets
    - Unquoted booleans/nulls (yes, no, true, false, null, None)
    - Single-quoted strings
    - Multiple candidate blocks in text

    Returns parsed object or None if truly unparseable.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # 1. Try markdown code block extraction
    code_block = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, re.DOTALL)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Find bracket-delimited candidates respecting string literals
    def extract_balanced(source: str, start_char: str, end_char: str) -> list[str]:
        candidates = []
        pos = 0
        while True:
            start = source.find(start_char, pos)
            if start < 0:
                break
            depth = 0
            in_str = False
            escape = False
            end = -1
            for i in range(start, len(source)):
                c = source[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == start_char:
                        depth += 1
                    elif c == end_char:
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
            if end != -1:
                candidates.append(source[start:end+1])
                pos = end + 1
            else:
                pos = start + 1
        return candidates

    def repair_json(s: str) -> str:
        # Remove trailing commas before } or ]
        s = re.sub(r",\s*([\}\]])", r"\1", s)
        # Unquoted boolean/null values
        s = re.sub(r":\s*yes\b", ": true", s, flags=re.IGNORECASE)
        s = re.sub(r":\s*no\b", ": false", s, flags=re.IGNORECASE)
        s = re.sub(r":\s*none\b", ": null", s, flags=re.IGNORECASE)
        s = re.sub(r":\s*true\b", ": true", s, flags=re.IGNORECASE)
        s = re.sub(r":\s*false\b", ": false", s, flags=re.IGNORECASE)
        return s

    parsed_candidates = []
    for start_c, end_c in [("{", "}"), ("[", "]")]:
        candidates = extract_balanced(text, start_c, end_c)
        for cand in candidates:
            obj = None
            try:
                obj = json.loads(cand)
            except (json.JSONDecodeError, ValueError):
                repaired = repair_json(cand)
                try:
                    obj = json.loads(repaired)
                except (json.JSONDecodeError, ValueError):
                    try:
                        obj = json.loads(repaired.replace("'", '"'))
                    except (json.JSONDecodeError, ValueError):
                        pass
            if obj is not None:
                parsed_candidates.append((len(cand), obj))

    if parsed_candidates:
        target_keys = {"nodes", "hypotheses", "concepts", "statement", "verdict", "checklist",
                       "questions", "domains", "queries", "synthesis", "abstraction", "gap", "match", "relevant"}
        for _, obj in parsed_candidates:
            if isinstance(obj, dict) and any(k in obj for k in target_keys):
                return obj
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (str, dict)):
                return obj
        parsed_candidates.sort(key=lambda x: x[0], reverse=True)
        return parsed_candidates[0][1]

    return None


def require_json(raw: str, default=None):
    """Parse LLM JSON output, returning default if unparseable."""
    result = parse_llm_json(raw)
    return result if result is not None else default


# ── Multi-model LLM calls ────────────────────────────────────────────────────

def llm_call(prompt: str, temperature: float = 0.7,
             model: str = None, system: str = None,
             role: str = "creative", format: str = None) -> str:
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
    kwargs = {"temperature": temperature, "num_predict": -1}
    chat_args = {
        "model": model,
        "messages": messages,
        "options": kwargs
    }
    if format:
        chat_args["format"] = format

    response = client.chat(**chat_args)
    if response.get('done_reason') not in (None, 'stop'):
        print(f"  [llm_call] non-stop done_reason={response.get('done_reason')!r} "
              f"role={role} model={model} -- response may be truncated")
    return response['message']['content'].strip()


def llm_json(prompt: str, temperature: float = 0.1,
             model: str = None, default=None,
             system: str = None, role: str = "precise") -> any:
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
        role=role,
        format="json"
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
        options={"temperature": temperature, "num_predict": -1}
    )
    if response.get('done_reason') not in (None, 'stop'):
        print(f"  [llm_chat] non-stop done_reason={response.get('done_reason')!r} "
              f"role={role} model={model} -- response may be truncated")
    return response['message']['content'].strip()


def stop_model(model_name: str):
    """Stop a specific Ollama model to free GPU VRAM immediately via API keep_alive=0."""
    import urllib.request
    import json
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            pass
    except Exception as e:
        import subprocess
        try:
            subprocess.run(["ollama", "stop", model_name], capture_output=True, timeout=10)
        except Exception:
            pass


def unload_all_models():
    """
    Unload all active Ollama models from GPU VRAM to ensure shared resources
    are freed immediately when tasks complete.
    """
    import subprocess
    try:
        res = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        lines = res.stdout.strip().splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    m = parts[0]
                    stop_model(m)
            print("  [unload_all_models] Released all models from GPU VRAM.")
    except Exception as e:
        print(f"  [unload_all_models] Warning: {e}")
