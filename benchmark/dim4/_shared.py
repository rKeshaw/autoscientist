"""
Shared utilities for Dimension 4 — Critic / System 2 Quality benchmarks.

Extends the dim3 shared graph with additional domains so the Critic has
a richer knowledge base for novelty checks, verdict calibration, and
refinement evaluation.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from critic.critic import Critic, CandidateThought, Verdict, CriticLog


DEFAULT_MISSION = (
    "How do biological and artificial learning systems balance "
    "exploration with stability?"
)

# ── Extended corpus ──────────────────────────────────────────────────────────
# The base dim2/dim3 corpus covered: DNA, Thermodynamics, Natural selection,
# ANN, Game theory, Genetics, Epigenetics, Mutation.
#
# For dim4 we add domains that make novelty checking, cross-domain verdict
# evaluation, and refinement more interesting.

BASE_CORPUS = [
    {"id": "dna", "title": "DNA"},
    {"id": "thermodynamics", "title": "Thermodynamics"},
    {"id": "natural_selection", "title": "Natural selection"},
    {"id": "ann", "title": "Artificial neural network"},
    {"id": "game_theory", "title": "Game theory"},
    {"id": "genetics", "title": "Genetics"},
    {"id": "epigenetics", "title": "Epigenetics"},
    {"id": "mutation", "title": "Mutation"},
]

EXTRA_CORPUS = [
    {"id": "reinforcement_learning", "title": "Reinforcement learning"},
    {"id": "neuroscience", "title": "Neuroscience"},
    {"id": "information_theory", "title": "Information theory"},
    {"id": "complex_systems", "title": "Complex system"},
]

FULL_CORPUS = BASE_CORPUS + EXTRA_CORPUS

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "how", "if", "in", "into", "is", "it", "of", "on",
    "or", "should", "than", "that", "the", "to", "what", "when", "where",
    "which", "while", "with", "would",
}


# ── Wikipedia helper ─────────────────────────────────────────────────────────

def fetch_wikipedia(title: str) -> str:
    import requests

    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "format": "json",
    }
    resp = requests.get(
        api,
        params=params,
        timeout=20,
        headers={"User-Agent": "AutoScientist-Benchmark/4.0"},
    )
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")[:8000]
    return ""


# ── Graph paths ──────────────────────────────────────────────────────────────

def _shared_paths(dim: str = "dim4"):
    shared_dir = ROOT / "benchmark" / dim / "shared"
    return shared_dir, shared_dir / "brain.json", shared_dir / "embedding_index"


def prepare_shared_graph(dim: str = "dim4"):
    """Build an enriched graph for dim4 by starting from dim3/dim2 and adding
    the extra corpus domains."""
    from graph.brain import Brain, EdgeSource
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer

    shared_dir, brain_path, index_path = _shared_paths(dim)
    os.makedirs(shared_dir, exist_ok=True)

    # Try to start from an existing dim3 or dim2 graph
    brain, emb_index = _try_load_existing_graph()

    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)

    # Determine which articles still need ingestion
    existing_node_count = brain.node_count if hasattr(brain, 'node_count') else len(list(brain.graph.nodes))
    articles_to_ingest = EXTRA_CORPUS if existing_node_count > 10 else FULL_CORPUS

    for article in articles_to_ingest:
        print(f"  Ingesting: {article['title']}...")
        text = fetch_wikipedia(article["title"])
        if text:
            ingestor.ingest(text, source=EdgeSource.READING)
            time.sleep(1)

    brain.save(str(brain_path))
    emb_index.save(str(index_path))
    return brain, emb_index


def _try_load_existing_graph():
    """Try to load an existing graph from dim4 → dim3 → dim2, or create empty."""
    from graph.brain import Brain
    from embedding_index import EmbeddingIndex

    for dim in ("dim4", "dim3", "dim2"):
        _, brain_path, index_path = _shared_paths(dim)
        if brain_path.exists() and Path(str(index_path) + ".json").exists():
            brain = Brain()
            brain.load(str(brain_path))
            emb_index = EmbeddingIndex.load(str(index_path))
            print(f"  Loaded existing graph from {dim}/shared/")
            return brain, emb_index

    return Brain(), EmbeddingIndex(dimension=384)


def load_or_build_shared_graph():
    """Load the dim4 shared graph, falling back to building it if needed."""
    from graph.brain import Brain
    from embedding_index import EmbeddingIndex

    _, brain_path, index_path = _shared_paths("dim4")
    if brain_path.exists() and Path(str(index_path) + ".json").exists():
        brain = Brain()
        brain.load(str(brain_path))
        emb_index = EmbeddingIndex.load(str(index_path))
        return brain, emb_index

    print("  Dim4 shared graph not found. Building enriched graph...")
    return prepare_shared_graph("dim4")


# ── Critic factory ───────────────────────────────────────────────────────────

def make_critic(
    mission: str = DEFAULT_MISSION,
    with_insight_buffer: bool = False,
):
    """Build a Critic with graph, embedding index, and optional insight buffer."""
    from observer.observer import Observer
    from insight_buffer import InsightBuffer

    brain, emb_index = load_or_build_shared_graph()

    if mission:
        brain.set_mission(mission)
        brain.complete_transition()

    insight_buffer = None
    if with_insight_buffer:
        insight_buffer = InsightBuffer(brain, embedding_index=emb_index)
        insight_buffer.pending = []  # start clean for benchmarks

    critic = Critic(
        brain,
        embedding_index=emb_index,
        insight_buffer=insight_buffer,
    )
    return critic, brain, emb_index, insight_buffer


# ── Candidate builders ───────────────────────────────────────────────────────

def make_high_stakes_candidate(
    claim: str,
    context: str = "",
    importance: float = 0.8,
    proposed_type: str = "synthesis",
    source_module: str = "thinker",
    crosses_domains: bool = False,
    contradicts_existing: bool = False,
) -> CandidateThought:
    return CandidateThought(
        claim=claim,
        source_module=source_module,
        proposed_type=proposed_type,
        importance=importance,
        context=context,
        crosses_domains=crosses_domains,
        contradicts_existing=contradicts_existing,
    )


def make_low_stakes_candidate(
    claim: str,
    context: str = "",
    importance: float = 0.3,
    proposed_type: str = "concept",
    source_module: str = "ingestor",
) -> CandidateThought:
    return CandidateThought(
        claim=claim,
        source_module=source_module,
        proposed_type=proposed_type,
        importance=importance,
        context=context,
    )


# ── Judge helper ─────────────────────────────────────────────────────────────

def judge_json(prompt: str, model: str, default: dict):
    from llm_utils import llm_call, require_json

    json_system = (
        "You are a strict evaluator. Respond ONLY with a valid JSON object. "
        "Do not include prose outside JSON, markdown fences, or comments."
    )
    raw = llm_call(
        prompt,
        temperature=0.1,
        model=model,
        system=json_system,
        role="precise",
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
        repair_prompt,
        temperature=0.0,
        model=model,
        system=json_system,
        role="precise",
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def content_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {token for token in tokens if token not in STOPWORDS}
