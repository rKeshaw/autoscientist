"""
Dimension 4 - Test 3: Novelty Check Accuracy
=============================================
Tests whether the Critic's _check_novelty() method correctly distinguishes
genuinely novel claims from redundant paraphrases of existing knowledge.

HYBRID APPROACH: Redundant claims are pulled directly from the actual graph
nodes (guaranteed to be there). Novel claims are handcrafted cross-domain
syntheses that definitely don't exist in the graph.
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_critic


# Novel claims — handcrafted cross-domain syntheses guaranteed not in graph
NOVEL_CLAIMS = [
    (
        "The topological invariants of gene regulatory networks may serve "
        "as a sufficient statistic for predicting evolutionary evolvability, "
        "analogous to how network depth predicts expressiveness in deep learning.",
        "Novel cross-domain synthesis — topology + evolvability + DL depth",
    ),
    (
        "Frustration in the epigenetic landscape can be quantified using "
        "the same energy-barrier metrics applied to spin glass models in "
        "statistical physics, suggesting a formal mapping between "
        "developmental canalization and thermodynamic ground states.",
        "Novel mapping — epigenetic landscape to spin glass physics",
    ),
    (
        "The critic-dreamer loop in cognitive architectures implements "
        "a biological analogue of the wake-sleep algorithm, where "
        "generative dreaming and discriminative criticism jointly refine "
        "an internal world model.",
        "Novel synthesis — cognitive architecture to wake-sleep algorithm",
    ),
    (
        "Reinforcement learning agents that use information-theoretic "
        "exploration bonuses based on prediction error show convergence "
        "properties that mirror the adaptive immune system's clonal "
        "selection mechanism.",
        "Novel analogy — RL exploration bonuses to immune clonal selection",
    ),
]


def _extract_redundant_claims(brain, count=4):
    """Pull actual node statements from the graph as ground-truth redundant claims."""
    candidates = []
    for nid in brain.graph.nodes:
        node = brain.get_node(nid)
        if node and node.get("statement"):
            stmt = node["statement"]
            # Filter for statements that are meaningful (not too short/long)
            if 30 < len(stmt) < 300:
                candidates.append((
                    stmt,
                    f"Redundant — exact node statement from graph (id={nid[:8]}...)"
                ))

    # Shuffle and pick a diverse subset
    random.seed(42)
    random.shuffle(candidates)
    return candidates[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim4/results/d4_novelty_check.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 3: Novelty Check Accuracy (graph-derived redundancy)")
    print("=" * 60)

    critic, brain, _, _ = make_critic()

    # Build test cases: novel (handcrafted) + redundant (from graph)
    redundant_claims = _extract_redundant_claims(brain, count=4)

    if len(redundant_claims) < 2:
        print("  WARNING: Graph too small — fewer than 2 usable node statements.")
        print("  Run prep_d4_graph.py first to build the shared graph.")

    cases = []
    for claim, desc in NOVEL_CLAIMS:
        cases.append((claim, True, desc))
    for claim, desc in redundant_claims:
        cases.append((claim, False, desc))

    evaluations = []
    correct_count = 0
    false_positives = 0   # redundant called novel
    false_negatives = 0   # novel called redundant

    for claim, expected_novel, description in cases:
        actual_novel = critic._check_novelty(claim)
        correct = actual_novel == expected_novel

        if correct:
            correct_count += 1
        elif expected_novel and not actual_novel:
            false_negatives += 1
        elif not expected_novel and actual_novel:
            false_positives += 1

        evaluations.append({
            "description": description,
            "claim": claim[:200],
            "expected_novel": expected_novel,
            "actual_novel": actual_novel,
            "correct": correct,
        })

        status = "✓" if correct else "✗"
        label = "NOVEL" if actual_novel else "REDUNDANT"
        expected_label = "NOVEL" if expected_novel else "REDUNDANT"
        print(f"  {status} {description[:60]}: expected={expected_label}, got={label}")
        time.sleep(0.2)

    n = max(len(cases), 1)
    novel_count = sum(1 for _, e, _ in cases if e)
    redundant_count = n - novel_count
    accuracy = correct_count / n
    false_positive_rate = false_positives / max(redundant_count, 1)
    false_negative_rate = false_negatives / max(novel_count, 1)

    passed = accuracy >= 0.75 and false_negative_rate <= 0.20

    report = {
        "test": "D4 - Novelty Check Accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"redundancy_source": "graph_nodes"},
        "summary": {
            "cases_evaluated": n,
            "novel_cases": novel_count,
            "redundant_cases": redundant_count,
            "correct_count": correct_count,
            "accuracy": round(accuracy, 3),
            "false_positive_rate": round(false_positive_rate, 3),
            "false_negative_rate": round(false_negative_rate, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAccuracy              : {accuracy:.2%}")
    print(f"False positive rate   : {false_positive_rate:.2%}")
    print(f"False negative rate   : {false_negative_rate:.2%}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
