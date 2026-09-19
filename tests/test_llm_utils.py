"""Unit tests for llm_utils parsing robustness."""
import pytest
from llm_utils import parse_llm_json, require_json

def test_parse_clean_json():
    text = '{"status": "ok", "count": 42}'
    data = parse_llm_json(text)
    assert data == {"status": "ok", "count": 42}

def test_parse_markdown_fenced():
    text = """Here is the result:
```json
{
  "name": "REM sleep",
  "score": 0.85
}
```
Hope that helps!"""
    data = parse_llm_json(text)
    assert data == {"name": "REM sleep", "score": 0.85}

def test_parse_braces_in_string():
    text = '{"claim": "Mapping {A} to {B} in system", "verdict": "ACCEPT"}'
    data = parse_llm_json(text)
    assert data["claim"] == "Mapping {A} to {B} in system"
    assert data["verdict"] == "ACCEPT"

def test_parse_trailing_commas():
    text = '{"items": [1, 2, 3, ], "extra": true, }'
    data = parse_llm_json(text)
    assert data["items"] == [1, 2, 3]
    assert data["extra"] is True

def test_parse_unquoted_none_null():
    text = '{"val": None, "other": null}'
    data = parse_llm_json(text)
    assert data["val"] is None
    assert data["other"] is None

def test_require_json_fallback():
    text = "Not json at all"
    fallback = {"default": True}
    assert require_json(text, default=fallback) == fallback
