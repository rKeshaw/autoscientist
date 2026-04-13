"""
Dimension 6 - Test 7: Time-to-Promotion
=======================================
Tests whether the insight buffer prunes stagnant pending pairs after repeated
failed evaluations on top of the inherited benchmark graph.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import DeterministicInsightBuffer, get_isolated_graph, resolve_node_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim6/results/d6_time_to_promotion.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 7: Time-to-Promotion")
    print("=" * 60)

    brain, emb_index = get_isolated_graph()
    tmp_dir = tempfile.mkdtemp(prefix="d6_t7_buffer_")
    buffer_path = os.path.join(tmp_dir, "insight_buffer.json")
    insight_buffer = DeterministicInsightBuffer(
        brain,
        embedding_index=emb_index,
        buffer_path=buffer_path,
        autoload=False,
        promote_when_shared_neighbors_at_least=99,
    )

    pair_a = resolve_node_id(
        brain,
        cluster="neuroscience",
        contains_all=["functional unit of the brain", "neuron"],
    )
    pair_b = resolve_node_id(
        brain,
        cluster="game_theory",
        contains_all=["nash equilibrium", "mixed-strategy equilibrium"],
    )
    younger_a = resolve_node_id(
        brain,
        cluster="evolutionary_biology",
        contains_all=["traits that confer higher fitness", "reproductive success"],
    )
    younger_b = resolve_node_id(
        brain,
        cluster="evolutionary_biology",
        contains_all=["fate of mutations within the population", "genetic drift"],
    )

    insight_buffer.add(
        pair_a,
        pair_b,
        0.46,
        "Weak cross-domain association from the inherited graph.",
        "Inherited graph pending edge.",
    )
    insight_buffer.add(
        younger_a,
        younger_b,
        0.47,
        "A younger pending real-graph pair that should remain after one more failed review.",
        "Inherited graph pending edge.",
    )

    for pending in insight_buffer.pending:
        if (
            (pending.node_a_id == pair_a and pending.node_b_id == pair_b)
            or (pending.node_a_id == pair_b and pending.node_b_id == pair_a)
        ):
            pending.times_evaluated = 9
        if (
            (pending.node_a_id == younger_a and pending.node_b_id == younger_b)
            or (pending.node_a_id == younger_b and pending.node_b_id == younger_a)
        ):
            pending.times_evaluated = 8

    result = insight_buffer.evaluate_all()
    pair_promoted = brain.graph.has_edge(pair_a, pair_b) or brain.graph.has_edge(pair_b, pair_a)
    pair_pending = any(
        (
            (pending.node_a_id == pair_a and pending.node_b_id == pair_b)
            or (pending.node_a_id == pair_b and pending.node_b_id == pair_a)
        )
        for pending in insight_buffer.pending
    )
    younger_pair_pending = any(
        (
            (pending.node_a_id == younger_a and pending.node_b_id == younger_b)
            or (pending.node_a_id == younger_b and pending.node_b_id == younger_a)
        )
        for pending in insight_buffer.pending
    )

    pair_pruned = not pair_promoted and not pair_pending
    passed = (
        pair_pruned
        and younger_pair_pending
        and result.get("pruned", 0) == 1
        and result.get("remaining", 0) == 1
    )

    summary = {
        "pruned_count": result.get("pruned", 0),
        "remaining_count": result.get("remaining", 0),
        "pair_pruned": pair_pruned,
        "younger_pair_still_pending": younger_pair_pending,
        "PASS": passed,
    }

    report_json = {
        "test": "D6 - Time to Promotion",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"graph_source": "benchmark/dim4/shared/brain.json"},
        "summary": summary,
        "details": {"buffer_eval_result": result},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    verdict = "PASS" if passed else "FAIL"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
