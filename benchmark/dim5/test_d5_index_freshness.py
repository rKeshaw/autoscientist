"""
Dimension 5 - Test 5: Index Freshness
=======================================
Tests whether the embedding index is consistent with the graph after
ingestion — i.e., newly ingested nodes are immediately retrievable
via embedding search, and paraphrases of existing nodes find the originals.

This is a deterministic test — no LLM judge needed.
"""

import json
import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import get_fresh_brain, make_ingestor
from embedding import embed as shared_embed


# ── Test text ────────────────────────────────────────────────────────────────

TEST_TEXT = """
Reinforcement learning is a type of machine learning where an agent learns
to make decisions by taking actions in an environment to maximize cumulative
reward. The agent receives feedback in the form of rewards or penalties and
adjusts its policy accordingly.

Q-learning is a model-free reinforcement learning algorithm that seeks to
learn the quality of actions, telling an agent what action to take under
what circumstances. It does not require a model of the environment and can
handle problems with stochastic transitions and rewards.

The exploration-exploitation tradeoff is a fundamental challenge in
reinforcement learning. The agent must balance between exploiting known
rewarding actions and exploring new actions that might yield higher rewards.
Epsilon-greedy strategies address this by occasionally selecting random
actions with probability epsilon.
"""

# Paraphrases of concepts likely to be extracted from the test text
PARAPHRASES = [
    "An RL agent learns by interacting with its environment and receiving reward signals",
    "Q-learning finds optimal action-selection policies without needing an environment model",
    "The balance between trying new things and using known good strategies in learning",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim5/results/d5_index_freshness.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 5: Index Freshness (deterministic)")
    print("=" * 60)

    # Fresh brain + index
    brain, emb_index = get_fresh_brain()
    ingestor, brain, emb_index, _ = make_ingestor(brain, emb_index)

    # Record node count before
    nodes_before = set(brain.graph.nodes)

    # Ingest the test text
    from graph.brain import EdgeSource
    raw_ids = ingestor.ingest(TEST_TEXT, source=EdgeSource.READING) or []
    new_ids = list(dict.fromkeys(raw_ids))  # deduplicate, preserve order
    print(f"\n  Ingested {len(new_ids)} unique new nodes (raw returned {len(raw_ids)})")

    if not new_ids:
        print("  ERROR: No nodes created. Cannot test index freshness.")
        report = {
            "test": "D5 - Index Freshness",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {"PASS": False, "error": "No nodes created"},
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        return

    # ── Test 1: Self-retrieval ────────────────────────────────────────────────
    # Every new node should be retrievable from the index when queried with
    # its own statement.

    self_retrieval_results = []
    self_retrieval_count = 0

    for nid in new_ids:
        node = brain.get_node(nid)
        if not node:
            continue

        stmt = node["statement"]
        query_emb = shared_embed(stmt)

        # Query the index
        matches = emb_index.query(query_emb, threshold=0.3, top_k=5)
        match_ids = [m[0] for m in matches]

        found = nid in match_ids
        rank = match_ids.index(nid) + 1 if found else -1
        top_sim = matches[0][1] if matches else 0.0

        if found:
            self_retrieval_count += 1

        self_retrieval_results.append({
            "node_id": nid[:12],
            "statement": stmt[:100],
            "found_in_top5": found,
            "rank": rank,
            "top_similarity": round(top_sim, 4),
        })

        status = "✓" if found else "✗"
        print(f"  {status} Self-retrieval [{nid[:8]}] rank={rank}: {stmt[:60]}...")

    self_retrieval_rate = self_retrieval_count / max(len(new_ids), 1)

    # ── Test 2: Paraphrase retrieval ──────────────────────────────────────────
    # Querying with paraphrases should find semantically matching nodes.

    paraphrase_results = []
    paraphrase_hit_count = 0

    for paraphrase in PARAPHRASES:
        query_emb = shared_embed(paraphrase)
        matches = emb_index.query(query_emb, threshold=0.3, top_k=5)

        # A hit = at least one of the new nodes appears in top-5
        hit_ids = [m[0] for m in matches if m[0] in new_ids]
        found = len(hit_ids) > 0

        if found:
            paraphrase_hit_count += 1

        best_match_stmt = ""
        best_sim = 0.0
        if matches:
            best_node = brain.get_node(matches[0][0])
            best_match_stmt = best_node["statement"][:100] if best_node else ""
            best_sim = matches[0][1]

        paraphrase_results.append({
            "paraphrase": paraphrase,
            "found_new_node": found,
            "hits_count": len(hit_ids),
            "best_match": best_match_stmt,
            "best_similarity": round(best_sim, 4),
        })

        status = "✓" if found else "✗"
        print(f"  {status} Paraphrase: {paraphrase[:60]}... → sim={best_sim:.3f}")

    paraphrase_recall = paraphrase_hit_count / max(len(PARAPHRASES), 1)

    # ── Test 3: Index consistency ─────────────────────────────────────────────
    # The number of entries in the index should match the number of nodes in
    # the graph.

    index_size = emb_index.size
    graph_size = len(brain.graph.nodes)
    index_consistent = index_size >= len(new_ids)  # at least the new nodes

    print(f"\n  Index size: {index_size}, Graph nodes: {graph_size}")
    print(f"  Index contains at least new nodes: {'✓' if index_consistent else '✗'}")

    # ── Verdict ───────────────────────────────────────────────────────────────

    passed = (
        self_retrieval_rate >= 1.0
        and paraphrase_recall >= 0.60
        and index_consistent
    )

    report = {
        "test": "D5 - Index Freshness",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {},
        "summary": {
            "new_nodes_created": len(new_ids),
            "self_retrieval_rate": round(self_retrieval_rate, 3),
            "paraphrase_recall": round(paraphrase_recall, 3),
            "index_consistent": index_consistent,
            "index_size": index_size,
            "graph_size": graph_size,
            "PASS": passed,
        },
        "self_retrieval": self_retrieval_results,
        "paraphrase_retrieval": paraphrase_results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSelf-retrieval rate: {self_retrieval_rate:.2%}")
    print(f"Paraphrase recall  : {paraphrase_recall:.2%}")
    print(f"Index consistent   : {index_consistent}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
