"""
Shared utilities for Dimension 6 — Consolidation & Insight Buffer benchmarks.

This suite mixes semantic generation benchmarks (synthesis / abstraction / gap
inference) with maintenance-path checks (contradiction upkeep, decay, and
delayed insight promotion), all anchored to the inherited shared benchmark
graph rather than synthetic standalone brains.
"""

import copy
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MISSION = (
    "How do biological and artificial learning systems balance "
    "exploration with stability?"
)


def _shared_paths(dim: str = "dim4"):
    shared_dir = ROOT / "benchmark" / dim / "shared"
    return shared_dir, shared_dir / "brain.json", shared_dir / "embedding_index"


def load_shared_graph():
    """
    Load the inherited benchmark graph.

    Dimension 5 reused the dim4 shared graph rather than building a new shared
    corpus, so D6 should inherit from the same artifact.
    """
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
    print(
        f"  Loaded shared graph: {len(brain.graph.nodes)} nodes, "
        f"{len(brain.graph.edges)} edges"
    )
    return brain, emb_index


def get_isolated_graph(mission: str | None = None):
    """Return a deep-copied working graph so each D6 test stays isolated."""
    brain, emb_index = load_shared_graph()
    brain_copy = copy.deepcopy(brain)
    index_copy = copy.deepcopy(emb_index)

    if mission:
        brain_copy.set_mission(mission)
        brain_copy.complete_transition()

    return brain_copy, index_copy


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def resolve_node_id(
    brain,
    *,
    contains_all: list[str],
    cluster: str | None = None,
    node_type: str | None = None,
    exclude_ids: set[str] | None = None,
):
    """
    Find a node in the inherited graph by stable content snippets.

    The dim4 shared graph is relatively stable, but exact IDs may change if the
    prep step is rebuilt. Content-based selectors are therefore safer than
    hardcoded node IDs.
    """
    exclude_ids = exclude_ids or set()
    lowered_terms = [term.lower() for term in contains_all]
    for nid, data in brain.all_nodes():
        if nid in exclude_ids:
            continue
        if cluster and data.get("cluster") != cluster:
            continue
        if node_type and data.get("node_type") != node_type:
            continue
        statement = normalize_text(data.get("statement", ""))
        if all(term in statement for term in lowered_terms):
            return nid
    raise KeyError(
        f"Could not resolve node selector cluster={cluster!r} "
        f"terms={contains_all!r}"
    )


def resolve_suite_nodes(brain, selectors: list[dict]) -> list[str]:
    resolved = []
    used = set()
    for selector in selectors:
        nid = resolve_node_id(
            brain,
            contains_all=selector["contains_all"],
            cluster=selector.get("cluster"),
            node_type=selector.get("node_type"),
            exclude_ids=used,
        )
        resolved.append(nid)
        used.add(nid)
    return resolved


def pairwise_similarity(consolidator, node_ids: list[str]) -> float:
    pair_sims = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            emb_i = consolidator._get_node_embedding(node_ids[i])
            emb_j = consolidator._get_node_embedding(node_ids[j])
            if emb_i is None or emb_j is None:
                continue
            pair_sims.append(consolidator._cosine(emb_i, emb_j))
    if not pair_sims:
        return 0.0
    return sum(pair_sims) / len(pair_sims)


@contextmanager
def override_models(model: str | None = None, critic_model: str | None = None):
    """
    Temporarily override role-based model routing used by runtime modules.

    Consolidator currently calls `llm_call(..., role="precise")`, so semantic
    D6 tests need a clean way to point that generation path at the requested
    benchmark model.
    """
    from config import MODELS

    attrs = ("CREATIVE", "PRECISE", "REASONING", "CRITIC")
    old = {attr: getattr(MODELS, attr) for attr in attrs}
    try:
        if model:
            MODELS.CREATIVE = model
            MODELS.PRECISE = model
            MODELS.REASONING = model
        if critic_model:
            MODELS.CRITIC = critic_model
        elif model:
            MODELS.CRITIC = model
        yield
    finally:
        for attr, value in old.items():
            setattr(MODELS, attr, value)


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


class DeterministicInsightBuffer:
    """
    Thin wrapper around InsightBuffer behavior for deterministic benchmark cases.

    The delayed-insight mechanics should be benchmarked as graph-state logic,
    not as a side effect of whatever embedding model or LLM happens to be
    configured locally.
    """

    def __new__(
        cls,
        *args,
        **kwargs,
    ):
        from insight_buffer import InsightBuffer

        class _Buffer(InsightBuffer):
            def __init__(
                self,
                *buffer_args,
                promote_when_shared_neighbors_at_least: int = 2,
                promotion_type: str = "supports",
                **buffer_kwargs,
            ):
                super().__init__(*buffer_args, **buffer_kwargs)
                self.promote_when_shared_neighbors_at_least = (
                    promote_when_shared_neighbors_at_least
                )
                self.promotion_type = promotion_type

            def _current_similarity(self, pair):
                return pair.original_similarity

            def _llm_evaluate(self, pair, node_a, node_b):
                shared = self._shared_neighbor_count(pair)
                if shared >= self.promote_when_shared_neighbors_at_least:
                    return {
                        "connected": True,
                        "type": self.promotion_type,
                        "narration": (
                            "Shared downstream context now provides enough "
                            "evidence to promote this delayed connection."
                        ),
                        "confidence": 0.78,
                    }
                return {"connected": False}

            def _llm_evaluate_node(self, pair, new_context):
                return {"accepted": False}

        return _Buffer(*args, **kwargs)
