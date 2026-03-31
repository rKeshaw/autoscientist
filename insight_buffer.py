"""
Insight Buffer — Delayed insight mechanism for near-miss connections.

When the system finds two ideas that are close but below the edge threshold
(e.g., similarity 0.45-0.59 when threshold is 0.60), it saves them here.
Each consolidation cycle, the buffer is re-evaluated — if new knowledge has
arrived that strengthens the connection, the pair gets promoted to a real edge.

This mimics the "shower insight" phenomenon: you see fact A on day 1 and
fact B on day 5, and on day 7 you suddenly realize they're connected.
"""

import json
import time
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from graph.brain import Brain, Edge, EdgeType, EdgeSource
from config import THRESHOLDS
from embedding import embed as shared_embed
from llm_utils import llm_call, require_json

# ── Config ────────────────────────────────────────────────────────────────────

BUFFER_PATH        = "data/insight_buffer.json"
BUFFER_LOW         = 0.45    # minimum similarity to enter buffer
MAX_EVALUATIONS    = 10      # prune after this many re-evaluations without improvement
MAX_BUFFER_SIZE    = 200     # hard cap on buffer size
PROMOTION_BOOST    = 0.10    # how much graph context can add to similarity score
NEIGHBOR_BOOST     = 0.05    # bonus per shared neighbor

# ── Prompt for re-evaluation ──────────────────────────────────────────────────

REEVALUATE_PROMPT = """You are examining two ideas that were previously noted as potentially
related but the connection wasn't strong enough to confirm.

Idea A: {node_a}
Idea B: {node_b}

Original similarity when first seen: {original_sim:.2f}
Times re-evaluated so far: {eval_count}

New context — ideas they are each now connected to:
A's neighbors: {a_neighbors}
B's neighbors: {b_neighbors}

In light of the new context, is there now a meaningful relationship between A and B?

Respond with a JSON object:
{{
  "connected": true or false,
  "type": one of ["supports", "causes", "contradicts", "analogy", "associated"],
  "analogy_depth": if type is "analogy", one of ["surface", "structural", "isomorphism"] — else omit,
  "narration": "one sentence explaining the connection",
  "confidence": a float 0.0-1.0 (how certain are you this link exists?)
}}

Confidence rubric:
- 0.1-0.3: Speculative — might be connected but you can't clearly articulate why
- 0.4-0.6: Plausible — the new context reveals a reasonable link
- 0.7-0.9: Strong — the connection is now clearly supported by the graph context
- 1.0: Definitive — the relationship is obvious given the neighbors

If not connected: {{"connected": false}}

Respond ONLY with JSON. No preamble.
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PendingInsight:
    node_a_id:          str
    node_b_id:          str
    original_similarity: float
    first_seen:         float  = field(default_factory=time.time)
    times_evaluated:    int    = 0
    last_eval_time:     float  = 0.0
    best_score:         float  = 0.0   # highest combined score seen
    promoted:           bool   = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingInsight":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Insight Buffer ────────────────────────────────────────────────────────────

class InsightBuffer:
    def __init__(self, brain: Brain, embedding_index=None):
        self.brain   = brain
        self.index   = embedding_index
        self.pending: list[PendingInsight] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(BUFFER_PATH, "r") as f:
                data = json.load(f)
            self.pending = [PendingInsight.from_dict(d) for d in data]
        except (FileNotFoundError, json.JSONDecodeError):
            self.pending = []

    def save(self):
        os.makedirs(os.path.dirname(BUFFER_PATH) or ".", exist_ok=True)
        with open(BUFFER_PATH, "w") as f:
            json.dump([p.to_dict() for p in self.pending], f, indent=2)

    # ── Add near-miss pairs ───────────────────────────────────────────────────

    def add(self, node_a_id: str, node_b_id: str, similarity: float):
        """Add a near-miss pair to the buffer."""
        if similarity < BUFFER_LOW:
            return  # too weak even for buffer

        # Check for duplicates
        for p in self.pending:
            if ((p.node_a_id == node_a_id and p.node_b_id == node_b_id) or
                (p.node_a_id == node_b_id and p.node_b_id == node_a_id)):
                # Update if this observation is stronger
                if similarity > p.original_similarity:
                    p.original_similarity = similarity
                return

        # Hard cap
        if len(self.pending) >= MAX_BUFFER_SIZE:
            # Drop the weakest
            self.pending.sort(key=lambda p: p.best_score or p.original_similarity)
            self.pending.pop(0)

        self.pending.append(PendingInsight(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            original_similarity=similarity
        ))

    @property
    def size(self) -> int:
        return len(self.pending)

    # ── Re-evaluation ─────────────────────────────────────────────────────────

    def evaluate_all(self) -> dict:
        """
        Re-evaluate all pending pairs. Returns stats dict.
        Called once per consolidation cycle.
        """
        if not self.pending:
            return {"evaluated": 0, "promoted": 0, "pruned": 0}

        promoted = 0
        pruned   = 0
        to_remove = []

        for i, pair in enumerate(self.pending):
            if pair.promoted:
                to_remove.append(i)
                continue

            # Check if both nodes still exist
            node_a = self.brain.get_node(pair.node_a_id)
            node_b = self.brain.get_node(pair.node_b_id)
            if not node_a or not node_b:
                to_remove.append(i)
                pruned += 1
                continue

            # Check if edge already exists (added by another mechanism)
            if (self.brain.graph.has_edge(pair.node_a_id, pair.node_b_id) or
                    self.brain.graph.has_edge(pair.node_b_id, pair.node_a_id)):
                to_remove.append(i)
                continue

            # Compute current similarity
            current_sim = self._current_similarity(pair)

            # Count shared neighbors as context boost
            shared = self._shared_neighbor_count(pair)
            context_score = current_sim + (shared * NEIGHBOR_BOOST)

            pair.times_evaluated += 1
            pair.last_eval_time = time.time()
            pair.best_score = max(pair.best_score, context_score)

            # If context score now exceeds threshold → LLM evaluation
            if context_score >= THRESHOLDS.WEAK_EDGE:
                result = self._llm_evaluate(pair, node_a, node_b)
                if result and result.get("connected"):
                    self._promote(pair, result)
                    to_remove.append(i)
                    promoted += 1
                    print(f"    ★ Promoted delayed insight: "
                          f"{node_a['statement'][:40]}... ↔ "
                          f"{node_b['statement'][:40]}...")
                    continue

            # Prune stale pairs
            if pair.times_evaluated >= MAX_EVALUATIONS:
                to_remove.append(i)
                pruned += 1

        # Remove in reverse order to maintain indices
        for i in sorted(to_remove, reverse=True):
            if i < len(self.pending):
                self.pending.pop(i)

        self.save()
        return {
            "evaluated": len(self.pending) + len(to_remove),
            "promoted":  promoted,
            "pruned":    pruned,
            "remaining": len(self.pending)
        }

    def _current_similarity(self, pair: PendingInsight) -> float:
        """Re-compute embedding similarity (may have changed if nodes were enriched)."""
        if self.index:
            emb_a = self.index.get_embedding(pair.node_a_id)
            emb_b = self.index.get_embedding(pair.node_b_id)
            if emb_a is not None and emb_b is not None:
                return float(np.dot(emb_a, emb_b))

        # Fallback: re-embed from statements
        node_a = self.brain.get_node(pair.node_a_id)
        node_b = self.brain.get_node(pair.node_b_id)
        if node_a and node_b:
            emb_a = shared_embed(node_a['statement'])
            emb_b = shared_embed(node_b['statement'])
            return float(np.dot(emb_a, emb_b))

        return pair.original_similarity

    def _shared_neighbor_count(self, pair: PendingInsight) -> int:
        """Count how many nodes are neighbors of BOTH A and B."""
        try:
            neighbors_a = set(self.brain.graph.neighbors(pair.node_a_id))
            neighbors_b = set(self.brain.graph.neighbors(pair.node_b_id))
            return len(neighbors_a & neighbors_b)
        except Exception:
            return 0

    def _get_neighbor_summaries(self, node_id: str, max_n: int = 5) -> str:
        """Get brief summaries of a node's neighbors for LLM context."""
        try:
            neighbors = list(self.brain.graph.neighbors(node_id))[:max_n]
            summaries = []
            for nid in neighbors:
                node = self.brain.get_node(nid)
                if node:
                    stmt = node['statement'][:80]
                    summaries.append(f"- {stmt}")
            return "\n".join(summaries) if summaries else "None yet"
        except Exception:
            return "None yet"

    def _llm_evaluate(self, pair: PendingInsight, node_a: dict,
                      node_b: dict) -> dict:
        """Ask LLM whether the pair is now connected given new context."""
        prompt = REEVALUATE_PROMPT.format(
            node_a=node_a['statement'],
            node_b=node_b['statement'],
            original_sim=pair.original_similarity,
            eval_count=pair.times_evaluated,
            a_neighbors=self._get_neighbor_summaries(pair.node_a_id),
            b_neighbors=self._get_neighbor_summaries(pair.node_b_id),
        )
        raw = llm_call(prompt, temperature=0.2, role="precise")
        return require_json(raw, default={})

    def _promote(self, pair: PendingInsight, result: dict):
        """Create a real edge from a promoted insight."""
        raw_type = result.get("type", "associated")
        confidence = result.get("confidence", 0.5)

        if raw_type == "analogy":
            depth = result.get("analogy_depth", "structural")
            self.brain.add_analogy_edge(
                pair.node_a_id, pair.node_b_id, depth,
                result.get("narration", "Delayed insight — connection emerged over time"),
                EdgeSource.CONSOLIDATION
            )
        else:
            try:
                etype = EdgeType(raw_type)
            except ValueError:
                etype = EdgeType.ASSOCIATED

            edge = Edge(
                type       = etype,
                narration  = result.get("narration",
                    "Delayed insight — connection emerged over time"),
                weight     = 0.5 + (confidence * 0.3),
                confidence = confidence,
                source     = EdgeSource.CONSOLIDATION
            )
            self.brain.add_edge(pair.node_a_id, pair.node_b_id, edge)

        pair.promoted = True

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "pending":        len(self.pending),
            "avg_similarity": (
                sum(p.original_similarity for p in self.pending) / len(self.pending)
                if self.pending else 0
            ),
            "avg_evaluations": (
                sum(p.times_evaluated for p in self.pending) / len(self.pending)
                if self.pending else 0
            ),
            "oldest_days": (
                (time.time() - min(p.first_seen for p in self.pending)) / 86400
                if self.pending else 0
            )
        }
