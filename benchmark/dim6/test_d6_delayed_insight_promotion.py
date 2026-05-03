"""
Dimension 6 - Test 6: Delayed Insight Promotion
===============================================
Tests whether the insight buffer promotes a deferred near-miss pair when the
inherited benchmark graph already contains enough shared downstream context to
justify promotion, while leaving a weak real pair pending.
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
        default="benchmark/dim6/results/d6_delayed_insight_promotion.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 6: Delayed Insight Promotion")
    print("=" * 60)

    brain, emb_index = get_isolated_graph()
    tmp_dir = tempfile.mkdtemp(prefix="d6_t6_buffer_")
    buffer_path = os.path.join(tmp_dir, "insight_buffer.json")
    insight_buffer = DeterministicInsightBuffer(
        brain,
        embedding_index=emb_index,
        buffer_path=buffer_path,
        autoload=False,
        promote_when_shared_neighbors_at_least=2,
        promotion_type="supports",
    )

    pair_a = resolve_node_id(
        brain,
        cluster="evolutionary_biology",
        contains_all=["traits that confer higher fitness", "reproductive success"],
    )
    pair_b = resolve_node_id(
        brain,
        cluster="evolutionary_biology",
        contains_all=["fate of mutations within the population", "genetic drift"],
    )
    pair_c = resolve_node_id(
        brain,
        cluster="neuroscience",
        contains_all=["functional unit of the brain", "neuron"],
    )
    pair_d = resolve_node_id(
        brain,
        cluster="game_theory",
        contains_all=["nash equilibrium", "mixed-strategy equilibrium"],
    )

    insight_buffer.add(
        pair_a,
        pair_b,
        0.55,
        "Potential delayed link around selection and mutation dynamics.",
        "Loaded from the inherited shared graph.",
    )
    insight_buffer.add(
        pair_c,
        pair_d,
        0.54,
        "A deliberately weak cross-domain near miss from the inherited graph.",
        "Loaded from the inherited shared graph.",
    )

    strong_shared_neighbors = insight_buffer._shared_neighbor_count(insight_buffer.pending[0])
    weak_shared_neighbors = insight_buffer._shared_neighbor_count(insight_buffer.pending[1])

    result = insight_buffer.evaluate_all()
    promoted_forward = brain.graph.has_edge(pair_a, pair_b)
    promoted_reverse = brain.graph.has_edge(pair_b, pair_a)
    weak_pair_promoted = brain.graph.has_edge(pair_c, pair_d) or brain.graph.has_edge(pair_d, pair_c)
    weak_pair_pending = any(
        (
            (pending.node_a_id == pair_c and pending.node_b_id == pair_d)
            or (pending.node_a_id == pair_d and pending.node_b_id == pair_c)
        )
        for pending in insight_buffer.pending
    )
    promoted_edge = None
    if promoted_forward:
        promoted_edge = brain.graph.get_edge_data(pair_a, pair_b)
    elif promoted_reverse:
        promoted_edge = brain.graph.get_edge_data(pair_b, pair_a)

    passed = (
        (promoted_forward or promoted_reverse)
        and result.get("promoted", 0) == 1
        and result.get("remaining", 0) == 1
        and strong_shared_neighbors >= 2
        and not weak_pair_promoted
        and weak_pair_pending
    )

    summary = {
        "promoted_count": result.get("promoted", 0),
        "remaining_count": result.get("remaining", 0),
        "strong_pair_shared_neighbors": strong_shared_neighbors,
        "weak_pair_shared_neighbors": weak_shared_neighbors,
        "strong_pair_promoted": promoted_forward or promoted_reverse,
        "weak_pair_left_pending": weak_pair_pending and not weak_pair_promoted,
        "PASS": passed,
    }

    report_json = {
        "test": "D6 - Delayed Insight Promotion",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"graph_source": "benchmark/dim4/shared/brain.json"},
        "summary": summary,
        "details": {
            "buffer_eval_result": result,
            "promoted_edge": promoted_edge,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    verdict = "PASS" if passed else "FAIL"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
