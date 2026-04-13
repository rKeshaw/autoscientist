"""
Dimension 5 - Test 7: Dedup Accuracy
======================================
Tests whether the Ingestor correctly merges semantic duplicates while
keeping distinct-but-related nodes separate.

Uses a fresh brain: first ingests a base set of statements, then ingests
variants (exact duplicates, paraphrases, distinct-but-related, unrelated)
and checks the merge decisions.
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

from _shared import get_fresh_brain


# ── Base statements (ingested first) ─────────────────────────────────────────

BASE_STATEMENTS = [
    "Natural selection favors organisms whose traits increase their reproductive fitness in a given environment.",
    "Entropy in a thermodynamic system measures the number of microscopic configurations that correspond to the system's macroscopic state.",
    "Backpropagation computes the gradient of the loss function with respect to each weight by applying the chain rule layer by layer.",
    "DNA stores genetic information in sequences of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "Game theory models strategic interactions between rational agents, each seeking to maximize their own utility.",
]

# ── Test cases: (statement, expected_action, description) ────────────────────
# expected_action: "merge" = should merge with an existing node
#                  "create" = should create a new node

TEST_CASES = [
    # Should MERGE: exact duplicate
    (
        "Natural selection favors organisms whose traits increase their reproductive fitness in a given environment.",
        "merge",
        "Exact duplicate of base statement #1",
    ),
    # Should MERGE: paraphrase
    (
        "Selection pressure in nature tends to preserve traits that enhance the ability of organisms to survive and reproduce.",
        "merge",
        "Paraphrase of base statement #1 (natural selection)",
    ),
    # Should MERGE: paraphrase of entropy
    (
        "Thermodynamic entropy quantifies the number of microstates consistent with the observed macrostate of a system.",
        "merge",
        "Paraphrase of base statement #2 (entropy)",
    ),
    # Should CREATE: distinct but related (same domain, different claim)
    (
        "Genetic drift causes random changes in allele frequencies, particularly in small populations, independent of selection pressure.",
        "create",
        "Distinct: genetic drift vs natural selection (different mechanism)",
    ),
    # Should CREATE: distinct but related
    (
        "The second law of thermodynamics states that the total entropy of an isolated system can never decrease over time.",
        "create",
        "Distinct: second law vs entropy definition (different claim)",
    ),
    # Should CREATE: distinct and unrelated domain
    (
        "The Krebs cycle is a series of chemical reactions used by aerobic organisms to release stored energy through oxidation of acetyl-CoA.",
        "create",
        "Distinct and unrelated: Krebs cycle (biochemistry)",
    ),
    # Should MERGE: paraphrase of backpropagation
    (
        "The backpropagation algorithm uses the chain rule of calculus to compute how much each network weight contributed to the output error.",
        "merge",
        "Paraphrase of base statement #3 (backpropagation)",
    ),
    # Should CREATE: distinct but related to backpropagation
    (
        "Gradient descent updates model parameters by moving in the direction opposite to the gradient of the loss function.",
        "create",
        "Distinct: gradient descent vs backpropagation (related but different)",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim5/results/d5_dedup_accuracy.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 7: Dedup Accuracy")
    print("=" * 60)

    from graph.brain import Brain, EdgeSource
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from insight_buffer import InsightBuffer

    # Fresh brain
    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    insight_buffer = InsightBuffer(brain, embedding_index=emb_index)
    insight_buffer.pending = []

    ingestor = Ingestor(
        brain, research_agenda=observer,
        embedding_index=emb_index, insight_buffer=insight_buffer,
    )

    # ── Phase 1: Ingest base statements ──────────────────────────────────────
    print("\n  Phase 1: Ingesting base statements")
    base_text = "\n\n".join(BASE_STATEMENTS)
    base_ids = ingestor.ingest(base_text, source=EdgeSource.READING) or []
    base_node_count = len(brain.graph.nodes)
    print(f"    Created {base_node_count} nodes from {len(BASE_STATEMENTS)} statements")

    # ── Phase 2: Ingest test cases one by one ────────────────────────────────
    print("\n  Phase 2: Testing merge decisions")

    evaluations = []
    correct_count = 0

    for statement, expected_action, description in TEST_CASES:
        nodes_before = set(brain.graph.nodes)
        test_ids = ingestor.ingest(statement, source=EdgeSource.READING) or []
        nodes_after = set(brain.graph.nodes)

        new_nodes = nodes_after - nodes_before
        # If no new nodes were created, the statement was merged
        # If test_ids contains an ID that was already in nodes_before, it was merged
        actual_action = "create"
        if not new_nodes:
            actual_action = "merge"
        elif test_ids and all(tid in nodes_before for tid in test_ids):
            actual_action = "merge"

        correct = actual_action == expected_action
        if correct:
            correct_count += 1

        evaluations.append({
            "description": description,
            "statement": statement[:150],
            "expected": expected_action,
            "actual": actual_action,
            "correct": correct,
            "new_nodes_created": len(new_nodes),
        })

        status = "✓" if correct else "✗"
        print(f"    {status} [{actual_action:6s}] {description}")

        time.sleep(0.2)

    # ── Metrics ──────────────────────────────────────────────────────────────

    n = len(TEST_CASES)
    accuracy = correct_count / max(n, 1)

    # Precision: of the merges we performed, how many were correct?
    actual_merges = [e for e in evaluations if e["actual"] == "merge"]
    correct_merges = [e for e in actual_merges if e["correct"]]
    merge_precision = len(correct_merges) / max(len(actual_merges), 1)

    # Recall: of the cases that should have been merged, how many were?
    expected_merges = [e for e in evaluations if e["expected"] == "merge"]
    caught_merges = [e for e in expected_merges if e["actual"] == "merge"]
    merge_recall = len(caught_merges) / max(len(expected_merges), 1)

    final_node_count = len(brain.graph.nodes)

    passed = merge_precision >= 0.80 and merge_recall >= 0.80

    report = {
        "test": "D5 - Dedup Accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {},
        "summary": {
            "base_node_count": base_node_count,
            "final_node_count": final_node_count,
            "test_cases": n,
            "correct_count": correct_count,
            "accuracy": round(accuracy, 3),
            "merge_precision": round(merge_precision, 3),
            "merge_recall": round(merge_recall, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nBase nodes    : {base_node_count}")
    print(f"Final nodes   : {final_node_count}")
    print(f"Accuracy      : {accuracy:.2%}")
    print(f"Merge precision: {merge_precision:.2%}")
    print(f"Merge recall   : {merge_recall:.2%}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
