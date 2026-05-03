"""
Shared utilities for Dimension 5 — Research & Reading Acquisition benchmarks.

Reuses the dim4 shared graph (genetics, thermodynamics, ML, evolutionary
biology, reinforcement learning, neuroscience, information theory, complex
systems).  Each test gets its own deepcopy of the graph to avoid cross-test
contamination.
"""

import copy
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MISSION = (
    "How do biological and artificial learning systems balance "
    "exploration with stability?"
)

# ── Graph loading ────────────────────────────────────────────────────────────

def _shared_paths(dim: str = "dim4"):
    shared_dir = ROOT / "benchmark" / dim / "shared"
    return shared_dir, shared_dir / "brain.json", shared_dir / "embedding_index"


def load_shared_graph():
    """Load the dim4 shared graph. Raises if not found."""
    from graph.brain import Brain
    from embedding_index import EmbeddingIndex

    _, brain_path, index_path = _shared_paths("dim4")
    if not brain_path.exists():
        raise FileNotFoundError(
            "Dim4 shared graph not found. Run benchmark/dim4/prep_d4_graph.py first."
        )

    brain = Brain()
    brain.load(str(brain_path))
    emb_index = EmbeddingIndex.load(str(index_path))
    print(f"  Loaded shared graph: {len(brain.graph.nodes)} nodes, "
          f"{len(brain.graph.edges)} edges")
    return brain, emb_index


def get_isolated_graph(mission: str = DEFAULT_MISSION):
    """Return a deep copy of the shared graph for isolated testing."""
    brain, emb_index = load_shared_graph()
    brain_copy = copy.deepcopy(brain)
    index_copy = copy.deepcopy(emb_index)

    if mission:
        brain_copy.set_mission(mission)
        brain_copy.complete_transition()

    return brain_copy, index_copy


def get_fresh_brain():
    """Return a completely fresh brain + index for tests that don't need prior knowledge."""
    from graph.brain import Brain
    from embedding_index import EmbeddingIndex

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    return brain, emb_index


# ── Module factories ─────────────────────────────────────────────────────────

def make_reader(brain=None, emb_index=None, mission=None):
    """Build a Reader with its own graph copy."""
    from reader.reader import Reader
    from observer.observer import Observer
    from ingestion.ingestor import Ingestor
    from insight_buffer import InsightBuffer

    if brain is None or emb_index is None:
        brain, emb_index = get_isolated_graph(mission or DEFAULT_MISSION)

    observer = Observer(brain)
    insight_buffer = InsightBuffer(brain, embedding_index=emb_index)
    insight_buffer.pending = []

    ingestor = Ingestor(
        brain, research_agenda=observer,
        embedding_index=emb_index, insight_buffer=insight_buffer,
    )

    reader = Reader(
        brain, observer=observer, notebook=None,
        ingestor=ingestor, embedding_index=emb_index,
        insight_buffer=insight_buffer,
    )

    return reader, brain, emb_index, observer


def make_researcher(brain=None, emb_index=None, mission=None, depth="standard"):
    """Build a Researcher with its own graph copy."""
    from researcher.researcher import Researcher
    from observer.observer import Observer
    from ingestion.ingestor import Ingestor
    from insight_buffer import InsightBuffer

    if brain is None or emb_index is None:
        brain, emb_index = get_isolated_graph(mission or DEFAULT_MISSION)

    observer = Observer(brain)
    insight_buffer = InsightBuffer(brain, embedding_index=emb_index)
    insight_buffer.pending = []

    ingestor = Ingestor(
        brain, research_agenda=observer,
        embedding_index=emb_index, insight_buffer=insight_buffer,
    )

    researcher = Researcher(
        brain, observer=observer, depth=depth,
        ingestor=ingestor, embedding_index=emb_index,
        insight_buffer=insight_buffer,
    )

    return researcher, brain, emb_index, observer


def make_ingestor(brain=None, emb_index=None, mission=None):
    """Build an Ingestor with its own graph copy."""
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from insight_buffer import InsightBuffer

    if brain is None or emb_index is None:
        brain, emb_index = get_isolated_graph(mission or DEFAULT_MISSION)

    observer = Observer(brain)
    insight_buffer = InsightBuffer(brain, embedding_index=emb_index)
    insight_buffer.pending = []

    ingestor = Ingestor(
        brain, research_agenda=observer,
        embedding_index=emb_index, insight_buffer=insight_buffer,
    )

    return ingestor, brain, emb_index, observer


# ── Judge helper ─────────────────────────────────────────────────────────────

def judge_json(prompt: str, model: str, default: dict):
    from llm_utils import llm_call, require_json

    json_system = (
        "You are a strict evaluator. Respond ONLY with a valid JSON object. "
        "Do not include prose outside JSON, markdown fences, or comments."
    )
    raw = llm_call(
        prompt, temperature=0.1, model=model,
        system=json_system, role="precise",
    )
    result = require_json(raw, default=None)
    if isinstance(result, dict):
        result.setdefault("_parse_failed", False)
        return result

    repair_prompt = (
        "Repair the following evaluator output into valid JSON matching this "
        f"schema:\n{json.dumps(default, indent=2)}\n\n"
        f"Evaluator output:\n{raw}\n\n"
        "Return ONLY the repaired JSON object."
    )
    repaired_raw = llm_call(
        repair_prompt, temperature=0.0, model=model,
        system=json_system, role="precise",
    )
    repaired = require_json(repaired_raw, default=None)
    if isinstance(repaired, dict):
        repaired.setdefault("_parse_failed", False)
        repaired["_repaired_from_raw"] = True
        return repaired

    fallback = dict(default)
    fallback["_parse_failed"] = True
    fallback["_raw_response"] = raw[:1000]
    return fallback
