"""
Dimension 6 - Test 5: Decay Calibration
=======================================
Tests whether Consolidator._apply_decay() correctly prunes weak edges,
preserves contradiction edges, keeps decay-exempt support, and decays a real
medium-strength edge on the inherited benchmark graph.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import consolidator.consolidator as consolidator_module
from _shared import get_isolated_graph, resolve_node_id
from consolidator.consolidator import ConsolidationReport, Consolidator
from graph.brain import Edge, EdgeSource, EdgeType


def _set_last_consolidation(path: str, days_ago: float):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(time.time() - (days_ago * 86400)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim6/results/d6_decay_calibration.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 5: Decay Calibration")
    print("=" * 60)

    brain, _ = get_isolated_graph()
    consolidator = Consolidator(brain)
    now = time.time()

    fresh_id = resolve_node_id(
        brain,
        cluster="neuroscience",
        contains_all=["functional unit of the brain", "neuron"],
    )
    stale_id = resolve_node_id(
        brain,
        cluster="genetics",
        contains_all=["modern genetics encompasses", "multiple scales"],
    )
    exempt_id = resolve_node_id(
        brain,
        cluster="thermodynamics",
        contains_all=["information-theoretic entropy", "classical thermodynamic entropy"],
    )
    medium_src = resolve_node_id(
        brain,
        cluster="thermodynamics",
        contains_all=["first law", "second law"],
    )
    medium_dst = resolve_node_id(
        brain,
        cluster="thermodynamics",
        contains_all=["determination of spontaneous processes", "state variables"],
    )
    contradiction_u = resolve_node_id(
        brain,
        cluster="genetics",
        contains_all=["traits acquired by an organism during its lifetime", "offspring"],
    )
    contradiction_v = resolve_node_id(
        brain,
        cluster="genetics",
        contains_all=["dna is the molecule responsible", "transmitting genetic information"],
    )

    brain.update_node(fresh_id, last_verified=now, source_quality=0.8)
    brain.update_node(stale_id, last_verified=now - (5 * 86400), source_quality=0.6)

    brain.add_edge(
        fresh_id,
        stale_id,
        Edge(
            type=EdgeType.ASSOCIATED,
            narration="Temporary weak speculative association that should decay away.",
            weight=0.05,
            source=EdgeSource.DREAM,
        ),
    )
    brain.add_edge(
        fresh_id,
        exempt_id,
        Edge(
            type=EdgeType.SUPPORTS,
            narration="A preserved support edge that should survive due to decay exemption.",
            weight=0.03,
            confidence=0.6,
            source=EdgeSource.CONSOLIDATION,
            decay_exempt=True,
        ),
    )

    medium_before = brain.get_edge(medium_src, medium_dst)
    if not medium_before:
        raise RuntimeError("Expected real medium support edge missing from shared graph.")
    medium_start_weight = float(medium_before["weight"])

    tmp_dir = tempfile.mkdtemp(prefix="d6_decay_")
    last_consolidation_path = os.path.join(tmp_dir, "last_consolidation.txt")
    _set_last_consolidation(last_consolidation_path, 400.0)

    old_path = consolidator_module.LAST_CONSOLIDATION_PATH
    try:
        consolidator_module.LAST_CONSOLIDATION_PATH = last_consolidation_path
        report = ConsolidationReport()
        consolidator._apply_decay(report)
    finally:
        consolidator_module.LAST_CONSOLIDATION_PATH = old_path

    weak_edge_survived = brain.graph.has_edge(fresh_id, stale_id)
    contradiction_survived = brain.graph.has_edge(contradiction_u, contradiction_v)
    exempt_edge_survived = brain.graph.has_edge(fresh_id, exempt_id)
    medium_edge_survived = brain.graph.has_edge(medium_src, medium_dst)
    medium_edge_data = brain.graph.get_edge_data(medium_src, medium_dst) if medium_edge_survived else None
    fresh_data = brain.get_node(fresh_id)
    stale_data = brain.get_node(stale_id)

    fresh_decayed = fresh_data["source_quality"] < 0.8
    stale_decayed = stale_data["source_quality"] < 0.6
    expected_medium_weight = max(
        0.01,
        medium_start_weight * (0.5 ** (400.0 / (1.0 / brain.decay_rate))),
    )
    medium_edge_calibrated = (
        medium_edge_survived
        and medium_edge_data is not None
        and math.isclose(
            medium_edge_data["weight"],
            expected_medium_weight,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    )
    timestamp_file_updated = os.path.exists(last_consolidation_path)

    passed = (
        not weak_edge_survived
        and contradiction_survived
        and exempt_edge_survived
        and medium_edge_calibrated
        and not fresh_decayed
        and stale_decayed
        and timestamp_file_updated
    )

    summary = {
        "weak_edge_pruned": not weak_edge_survived,
        "contradiction_preserved": contradiction_survived,
        "decay_exempt_edge_preserved": exempt_edge_survived,
        "medium_edge_decayed_but_preserved": medium_edge_calibrated,
        "fresh_node_preserved": not fresh_decayed,
        "stale_node_decayed": stale_decayed,
        "edges_decayed_count": report.edges_decayed,
        "timestamp_file_updated": timestamp_file_updated,
        "PASS": passed,
    }

    report_json = {
        "test": "D6 - Decay Calibration",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"graph_source": "benchmark/dim4/shared/brain.json"},
        "summary": summary,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    verdict = "PASS" if passed else "FAIL"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
