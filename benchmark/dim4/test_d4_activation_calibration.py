"""
Dimension 4 - Test 1: Activation Calibration
=============================================
Tests the Critic's laziness gate (needs_review) — whether it correctly
identifies high-stakes claims that need review and low-stakes claims
that should be bypassed.

This test is entirely deterministic — no LLM calls.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_critic, make_high_stakes_candidate, make_low_stakes_candidate
from critic.critic import CandidateThought


# ── Test cases ───────────────────────────────────────────────────────────────
# Each case: (candidate_kwargs, expected_needs_review, description)

CASES = [
    # === Should trigger review ===
    (
        {"claim": "Epigenetic memory implements a form of regularization.",
         "proposed_type": "synthesis", "importance": 0.8},
        True,
        "High-importance synthesis — ALWAYS_REVIEW_TYPES",
    ),
    (
        {"claim": "Chromatin insulation is structurally analogous to dropout regularization.",
         "proposed_type": "hypothesis", "importance": 0.9},
        True,
        "High-importance hypothesis — ALWAYS_REVIEW_TYPES",
    ),
    (
        {"claim": "DNA methylation maps onto weight freezing in neural networks.",
         "edge_type": "structural_analogy", "importance": 0.5},
        True,
        "Structural analogy edge — ALWAYS_REVIEW_TYPES",
    ),
    (
        {"claim": "Attractor dynamics and policy gradient are isomorphic.",
         "edge_type": "deep_isomorphism", "importance": 0.4},
        True,
        "Deep isomorphism edge — ALWAYS_REVIEW_TYPES",
    ),
    (
        {"claim": "Stochastic variation maps to gradient noise.",
         "proposed_type": "concept", "importance": 0.75},
        False,
        "High-importance concept — still bypassed (BYPASS_TYPES overrides importance)",
    ),
    (
        {"claim": "Selection pressure parallels loss function curvature.",
         "proposed_type": "concept", "importance": 0.5, "crosses_domains": True},
        True,
        "Cross-domain claim — crosses_domains triggers",
    ),
    (
        {"claim": "This directly contradicts the Central Dogma.",
         "proposed_type": "concept", "importance": 0.3, "contradicts_existing": True},
        True,
        "Contradicts existing knowledge — contradicts_existing triggers",
    ),

    # === Should bypass review ===
    (
        {"claim": "DNA is a double-helix molecule.",
         "proposed_type": "concept", "importance": 0.3},
        False,
        "Low-importance concept — BYPASS_TYPES",
    ),
    (
        {"claim": "Mutations can be beneficial or deleterious.",
         "proposed_type": "concept", "importance": 0.4},
        False,
        "Below ACTIVATION_THRESHOLD, concept type",
    ),
    (
        {"claim": "Gene A is associated with Gene B.",
         "edge_type": "associated", "importance": 0.5},
        False,
        "Associated edge — BYPASS_TYPES",
    ),
    (
        {"claim": "These two ideas share surface-level similarities.",
         "edge_type": "surface_analogy", "importance": 0.6},
        False,
        "Surface analogy edge — BYPASS_TYPES",
    ),
    (
        {"claim": "Thermodynamics studies energy transfer.",
         "proposed_type": "concept", "importance": 0.6},
        False,
        "Concept just below ACTIVATION_THRESHOLD (0.65)",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim4/results/d4_activation_calibration.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 1: Critic Activation Calibration (deterministic)")
    print("=" * 60)

    critic, _, _, _ = make_critic()

    evaluations = []
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for kwargs, expected, description in CASES:
        candidate = CandidateThought(**kwargs)
        actual = critic.needs_review(candidate)
        correct = actual == expected

        if expected and actual:
            true_positives += 1
        elif not expected and not actual:
            true_negatives += 1
        elif not expected and actual:
            false_positives += 1
        elif expected and not actual:
            false_negatives += 1

        evaluations.append({
            "description": description,
            "claim": kwargs.get("claim", ""),
            "proposed_type": kwargs.get("proposed_type", ""),
            "edge_type": kwargs.get("edge_type", ""),
            "importance": kwargs.get("importance", 0.7),
            "crosses_domains": kwargs.get("crosses_domains", False),
            "contradicts_existing": kwargs.get("contradicts_existing", False),
            "expected_needs_review": expected,
            "actual_needs_review": actual,
            "correct": correct,
        })

        status = "✓" if correct else "✗"
        print(f"  {status} {description}: expected={expected}, got={actual}")

    total = len(CASES)
    positive_count = sum(1 for _, e, _ in CASES if e)
    negative_count = total - positive_count
    accuracy = (true_positives + true_negatives) / max(total, 1)
    true_positive_rate = true_positives / max(positive_count, 1)
    true_negative_rate = true_negatives / max(negative_count, 1)

    passed = accuracy >= 0.90 and true_positive_rate >= 0.95

    report = {
        "test": "D4 - Activation Calibration",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"deterministic": True},
        "summary": {
            "cases_evaluated": total,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "accuracy": round(accuracy, 3),
            "true_positive_rate": round(true_positive_rate, 3),
            "true_negative_rate": round(true_negative_rate, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAccuracy             : {accuracy:.2%}")
    print(f"True Positive Rate   : {true_positive_rate:.2%}")
    print(f"True Negative Rate   : {true_negative_rate:.2%}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
