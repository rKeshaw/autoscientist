"""
Dimension 5 - Test 6: Predictive Processing Calibration
=========================================================
Tests whether the Ingestor's predictive processing mechanism correctly
modulates node importance based on prediction error (surprise).

Deterministic test — no LLM judge needed.

Benchmark level:
  - module-level
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


# ── Test cases ───────────────────────────────────────────────────────────────

# Case 1: prediction is highly aligned with the text → low surprise
ALIGNED_PREDICTION = (
    "This text will discuss reinforcement learning agents that learn from "
    "reward signals and optimize policies through interaction with environments."
)
ALIGNED_TEXT = """
Reinforcement learning is a paradigm in machine learning where an agent
learns optimal behavior through trial-and-error interaction with an
environment. The agent receives scalar reward signals after each action
and adjusts its policy to maximize cumulative future rewards.
"""

# Case 2: prediction is misaligned with the text → high surprise
MISALIGNED_PREDICTION = (
    "This text will discuss the chemical properties of noble gases and "
    "their electron configurations in the periodic table."
)
MISALIGNED_TEXT = """
Reinforcement learning is a paradigm in machine learning where an agent
learns optimal behavior through trial-and-error interaction with an
environment. The agent receives scalar reward signals after each action
and adjusts its policy to maximize cumulative future rewards.
"""

# Case 3: no prediction → importance stays at baseline
BASELINE_TEXT = """
Reinforcement learning is a paradigm in machine learning where an agent
learns optimal behavior through trial-and-error interaction with an
environment. The agent receives scalar reward signals after each action
and adjusts its policy to maximize cumulative future rewards.
"""


def _get_mean_importance(brain, node_ids):
    """Get the mean importance of a set of nodes."""
    importances = []
    for nid in node_ids:
        node = brain.get_node(nid)
        if node:
            importances.append(node.get("importance", 0.5))
    return np.mean(importances) if importances else 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim5/results/d5_predictive_processing.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 6: Predictive Processing Calibration (deterministic)")
    print("=" * 60)

    from graph.brain import EdgeSource

    results = {}

    # ── Case 1: Low surprise (aligned prediction) ────────────────────────────
    print("\n  Case 1: Aligned prediction → expect LOW surprise, DAMPENED importance")

    brain1, idx1 = get_fresh_brain()
    ing1, brain1, idx1, _ = make_ingestor(brain1, idx1)
    ids_aligned = ing1.ingest(
        ALIGNED_TEXT, source=EdgeSource.READING,
        prediction=ALIGNED_PREDICTION,
    ) or []
    mean_imp_aligned = _get_mean_importance(brain1, ids_aligned)
    print(f"    Nodes: {len(ids_aligned)}, Mean importance: {mean_imp_aligned:.3f}")

    # ── Case 2: High surprise (misaligned prediction) ────────────────────────
    print("\n  Case 2: Misaligned prediction → expect HIGH surprise, BOOSTED importance")

    brain2, idx2 = get_fresh_brain()
    ing2, brain2, idx2, _ = make_ingestor(brain2, idx2)
    ids_misaligned = ing2.ingest(
        MISALIGNED_TEXT, source=EdgeSource.READING,
        prediction=MISALIGNED_PREDICTION,
    ) or []
    mean_imp_misaligned = _get_mean_importance(brain2, ids_misaligned)
    print(f"    Nodes: {len(ids_misaligned)}, Mean importance: {mean_imp_misaligned:.3f}")

    # ── Case 3: Baseline (no prediction) ─────────────────────────────────────
    print("\n  Case 3: No prediction → expect BASELINE importance")

    brain3, idx3 = get_fresh_brain()
    ing3, brain3, idx3, _ = make_ingestor(brain3, idx3)
    ids_baseline = ing3.ingest(
        BASELINE_TEXT, source=EdgeSource.READING,
        prediction="",
    ) or []
    mean_imp_baseline = _get_mean_importance(brain3, ids_baseline)
    print(f"    Nodes: {len(ids_baseline)}, Mean importance: {mean_imp_baseline:.3f}")

    # ── Evaluation ───────────────────────────────────────────────────────────

    # The key invariant: misaligned importance > baseline > aligned importance
    surprise_ordering_correct = bool(
        mean_imp_misaligned > mean_imp_baseline > mean_imp_aligned
    )

    print(f"\n  Importance ordering:")
    print(f"    Aligned (low surprise):   {mean_imp_aligned:.3f}")
    print(f"    Baseline (no prediction): {mean_imp_baseline:.3f}")
    print(f"    Misaligned (high surprise): {mean_imp_misaligned:.3f}")
    print(f"    Full ordering correct: {'✓' if surprise_ordering_correct else '✗'}")

    passed = surprise_ordering_correct

    report = {
        "test": "D5 - Predictive Processing Calibration",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {},
        "summary": {
            "aligned_mean_importance": float(round(mean_imp_aligned, 4)),
            "baseline_mean_importance": float(round(mean_imp_baseline, 4)),
            "misaligned_mean_importance": float(round(mean_imp_misaligned, 4)),
            "full_ordering_correct": surprise_ordering_correct,
            "PASS": passed,
        },
        "cases": {
            "aligned": {
                "prediction": ALIGNED_PREDICTION,
                "nodes_created": len(ids_aligned),
                "mean_importance": float(round(mean_imp_aligned, 4)),
            },
            "misaligned": {
                "prediction": MISALIGNED_PREDICTION,
                "nodes_created": len(ids_misaligned),
                "mean_importance": float(round(mean_imp_misaligned, 4)),
            },
            "baseline": {
                "prediction": "",
                "nodes_created": len(ids_baseline),
                "mean_importance": float(round(mean_imp_baseline, 4)),
            },
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
