"""
Dimension 6 - Test 4: Contradiction Maintenance
===============================================
Tests whether Consolidator._contradiction_update() keeps real unresolved
tensions from the inherited benchmark graph salient by increasing the
importance of contradiction participants.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import get_isolated_graph, resolve_node_id
from consolidator.consolidator import ConsolidationReport, Consolidator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim6/results/d6_contradiction_maintenance.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 4: Contradiction Maintenance")
    print("=" * 60)

    brain, _ = get_isolated_graph()
    consolidator = Consolidator(brain)

    lamarck_id = resolve_node_id(
        brain,
        cluster="genetics",
        contains_all=[
            "traits acquired by an organism during its lifetime",
            "offspring",
        ],
    )
    dna_id = resolve_node_id(
        brain,
        cluster="genetics",
        contains_all=[
            "dna is the molecule responsible",
            "transmitting genetic information",
        ],
    )
    linkage_id = resolve_node_id(
        brain,
        cluster="genetics",
        contains_all=["sex-linked mutations", "chromosomes"],
    )
    fitness_id = resolve_node_id(
        brain,
        cluster="evolutionary_biology",
        contains_all=["traits that confer higher fitness", "reproductive success"],
    )
    control_id = resolve_node_id(
        brain,
        cluster="neuroscience",
        contains_all=["functional unit of the brain", "neuron"],
    )

    # Calibrate a few real contradiction participants so the check is precise.
    brain.update_node(lamarck_id, importance=0.98)
    brain.update_node(dna_id, importance=0.97)
    brain.update_node(linkage_id, importance=0.40)
    brain.update_node(fitness_id, importance=0.55)

    before_lamarck = brain.get_node(lamarck_id)["importance"]
    before_dna = brain.get_node(dna_id)["importance"]
    before_linkage = brain.get_node(linkage_id)["importance"]
    before_fitness = brain.get_node(fitness_id)["importance"]
    before_control = brain.get_node(control_id)["importance"]
    expected_contradiction_count = sum(
        1
        for _, _, data in brain.graph.edges(data=True)
        if data.get("type") == "contradicts"
    )

    report = ConsolidationReport()
    consolidator._contradiction_update(report)

    after_lamarck = brain.get_node(lamarck_id)["importance"]
    after_dna = brain.get_node(dna_id)["importance"]
    after_linkage = brain.get_node(linkage_id)["importance"]
    after_fitness = brain.get_node(fitness_id)["importance"]
    after_control = brain.get_node(control_id)["importance"]

    linkage_bumped = abs(after_linkage - min(1.0, before_linkage + 0.05)) < 1e-9
    fitness_bumped = abs(after_fitness - min(1.0, before_fitness + 0.05)) < 1e-9
    control_unchanged = abs(after_control - before_control) < 1e-9
    lamarck_capped = abs(after_lamarck - 1.0) < 1e-9
    dna_capped = abs(after_dna - 1.0) < 1e-9
    report_count_correct = report.contradictions_updated == expected_contradiction_count

    passed = (
        linkage_bumped
        and fitness_bumped
        and control_unchanged
        and lamarck_capped
        and dna_capped
        and report_count_correct
    )
    summary = {
        "contradictory_node_a_bumped": linkage_bumped,
        "contradictory_node_b_bumped": fitness_bumped,
        "control_node_unchanged": control_unchanged,
        "capped_node_a_capped_at_one": lamarck_capped,
        "capped_node_b_capped_at_one": dna_capped,
        "contradictions_updated": report.contradictions_updated,
        "PASS": passed,
    }

    report_json = {
        "test": "D6 - Contradiction Maintenance",
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
