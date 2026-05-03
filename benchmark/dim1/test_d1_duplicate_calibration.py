"""
Dimension 1 — Test 3: Duplicate Calibration
===========================================
Evaluates whether the deduplication path merges true paraphrases while
preserving distinct but semantically nearby scientific claims.

Unlike the duplicate-rate benchmark, this uses a frozen fixture set with:
  - positive duplicate paraphrases
  - hard negatives from the same domain
  - boundary cases that should stay separate

It exercises the real `Ingestor._process_statement(...)` dedup path,
including embedding lookup, thresholding, and node enrichment.

Pass criterion:
  - duplicate recall >= 80%
  - hard-negative specificity >= 85%
  - boundary specificity >= 67%
  - overall accuracy >= 80%

Benchmark level:
  - module-level

Usage:
    python benchmark/dim1/test_d1_duplicate_calibration.py \
        --judge-model <ollama-model-name> \
        --out benchmark/dim1/results/d1_duplicate_calibration.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from embedding import embed as shared_embed


CASES = [
    {
        "id": "dna_duplicate",
        "category": "duplicate",
        "anchor": (
            "DNA stores hereditary information in sequences of nucleotides "
            "that are copied during replication."
        ),
        "candidate": (
            "Hereditary information is encoded in DNA base sequences and "
            "preserved when those sequences are replicated."
        ),
        "expected_merge": True,
    },
    {
        "id": "selection_duplicate",
        "category": "duplicate",
        "anchor": (
            "Natural selection changes populations because heritable traits "
            "that improve reproductive success become more common."
        ),
        "candidate": (
            "Traits that increase reproductive success spread through a "
            "population under natural selection when those traits are heritable."
        ),
        "expected_merge": True,
    },
    {
        "id": "gradient_duplicate",
        "category": "duplicate",
        "anchor": (
            "Gradient descent trains neural networks by adjusting weights in "
            "the direction that reduces loss."
        ),
        "candidate": (
            "Neural network training uses gradient descent to update weights "
            "so the loss function decreases."
        ),
        "expected_merge": True,
    },
    {
        "id": "nash_duplicate",
        "category": "duplicate",
        "anchor": (
            "A Nash equilibrium is a strategy profile where no player can gain "
            "by changing strategy alone."
        ),
        "candidate": (
            "In Nash equilibrium, unilateral deviation does not improve any "
            "player's payoff."
        ),
        "expected_merge": True,
    },
    {
        "id": "entropy_duplicate",
        "category": "duplicate",
        "anchor": (
            "In an isolated system, entropy tends to increase over time."
        ),
        "candidate": (
            "Entropy ordinarily rises in isolated systems as time passes."
        ),
        "expected_merge": True,
    },
    {
        "id": "dna_hard_negative",
        "category": "hard_negative",
        "anchor": (
            "DNA replication preserves genetic information across cell divisions."
        ),
        "candidate": (
            "DNA transcription copies genetic information into RNA for gene "
            "expression."
        ),
        "expected_merge": False,
    },
    {
        "id": "deep_learning_hard_negative",
        "category": "hard_negative",
        "anchor": (
            "Gradient descent reduces training loss by updating neural network "
            "weights."
        ),
        "candidate": (
            "Dropout reduces overfitting by randomly masking units during "
            "training."
        ),
        "expected_merge": False,
    },
    {
        "id": "game_theory_hard_negative",
        "category": "hard_negative",
        "anchor": (
            "A Nash equilibrium is stable against unilateral deviation."
        ),
        "candidate": (
            "Pareto optimality means no participant can be improved without "
            "making someone else worse off."
        ),
        "expected_merge": False,
    },
    {
        "id": "thermo_hard_negative",
        "category": "hard_negative",
        "anchor": (
            "Entropy tends to increase in isolated systems."
        ),
        "candidate": (
            "Free energy determines whether a process is spontaneous at "
            "constant temperature and pressure."
        ),
        "expected_merge": False,
    },
    {
        "id": "dna_boundary",
        "category": "boundary",
        "anchor": (
            "DNA polymerase synthesizes new DNA strands only in the 5' to 3' "
            "direction."
        ),
        "candidate": (
            "DNA replication requires a primer because DNA polymerases cannot "
            "start synthesis de novo."
        ),
        "expected_merge": False,
    },
    {
        "id": "learning_boundary",
        "category": "boundary",
        "anchor": (
            "Backpropagation computes gradients efficiently through layered "
            "neural networks."
        ),
        "candidate": (
            "Gradient descent applies computed gradients to update neural "
            "network weights."
        ),
        "expected_merge": False,
    },
    {
        "id": "evolution_boundary",
        "category": "boundary",
        "anchor": (
            "Mutations introduce new heritable variation into populations."
        ),
        "candidate": (
            "Recombination reshuffles existing alleles into new combinations."
        ),
        "expected_merge": False,
    },
]


@contextmanager
def override_models(model: str | None = None):
    from config import MODELS

    attrs = ("CREATIVE", "PRECISE", "REASONING", "CRITIC")
    old = {attr: getattr(MODELS, attr) for attr in attrs}
    try:
        if model:
            MODELS.CREATIVE = model
            MODELS.PRECISE = model
            MODELS.REASONING = model
            MODELS.CRITIC = model
        yield
    finally:
        for attr, value in old.items():
            setattr(MODELS, attr, value)


def cosine_similarity(text_a: str, text_b: str) -> float:
    emb_a = shared_embed(text_a)
    emb_b = shared_embed(text_b)
    return float(emb_a @ emb_b)


def run_case(case: dict):
    from graph.brain import Brain, EdgeSource, NodeType
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)

    source = EdgeSource.READING
    anchor_id = ingestor._process_statement(
        case["anchor"], None, source, NodeType.CONCEPT
    )
    nodes_before = len(brain.all_nodes())
    candidate_id = ingestor._process_statement(
        case["candidate"], None, source, NodeType.CONCEPT
    )
    nodes_after = len(brain.all_nodes())

    merged = (
        anchor_id is not None and
        candidate_id == anchor_id and
        nodes_after == nodes_before
    )
    created_new_node = (
        anchor_id is not None and
        candidate_id is not None and
        candidate_id != anchor_id and
        nodes_after == nodes_before + 1
    )
    anchor_after = brain.get_node(anchor_id) if anchor_id else None
    enriched_statement = (anchor_after or {}).get("statement", "")
    enriched = " | " in enriched_statement

    if case["expected_merge"]:
        correct = merged and enriched
    else:
        correct = created_new_node

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_merge": case["expected_merge"],
        "observed_merge": merged,
        "created_new_node": created_new_node,
        "correct": correct,
        "anchor_id": anchor_id,
        "candidate_id": candidate_id,
        "nodes_before_candidate": nodes_before,
        "nodes_after_candidate": nodes_after,
        "pair_similarity": round(
            cosine_similarity(case["anchor"], case["candidate"]), 4
        ),
        "anchor": case["anchor"],
        "candidate": case["candidate"],
        "enriched_statement": enriched_statement if merged else "",
    }


def metric_fraction(results: list[dict], predicate) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if predicate(r)) / len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim1/results/d1_duplicate_calibration.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with override_models(args.judge_model):
        results = []
        print("=" * 60)
        print("PHASE 1: Running duplicate-calibration fixture cases")
        print("=" * 60)
        for i, case in enumerate(CASES, start=1):
            print(f"  [{i}/{len(CASES)}] {case['id']} ({case['category']})")
            result = run_case(case)
            results.append(result)
            observed = "MERGED" if result["observed_merge"] else "SEPARATE"
            expected = "MERGE" if case["expected_merge"] else "SEPARATE"
            status = "✓" if result["correct"] else "✗"
            print(
                f"      {status} expected={expected} observed={observed} "
                f"sim={result['pair_similarity']:.3f}"
            )

    duplicate_cases = [r for r in results if r["category"] == "duplicate"]
    hard_negative_cases = [r for r in results if r["category"] == "hard_negative"]
    boundary_cases = [r for r in results if r["category"] == "boundary"]
    nonduplicate_cases = [
        r for r in results if r["category"] in ("hard_negative", "boundary")
    ]

    duplicate_recall = metric_fraction(duplicate_cases, lambda r: r["correct"])
    hard_negative_specificity = metric_fraction(
        hard_negative_cases, lambda r: r["correct"]
    )
    boundary_specificity = metric_fraction(
        boundary_cases, lambda r: r["correct"]
    )
    nonduplicate_specificity = metric_fraction(
        nonduplicate_cases, lambda r: r["correct"]
    )
    overall_accuracy = metric_fraction(results, lambda r: r["correct"])

    false_merges = [
        r for r in results if (not r["expected_merge"]) and r["observed_merge"]
    ]
    missed_duplicates = [
        r for r in results if r["expected_merge"] and (not r["observed_merge"])
    ]

    passed = (
        duplicate_recall >= 0.80 and
        hard_negative_specificity >= 0.85 and
        boundary_specificity >= 0.67 and
        overall_accuracy >= 0.80
    )

    report = {
        "test": "D1 — Duplicate Calibration",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
            "duplicate_cases": len(duplicate_cases),
            "hard_negative_cases": len(hard_negative_cases),
            "boundary_cases": len(boundary_cases),
        },
        "summary": {
            "duplicate_recall": round(duplicate_recall, 3),
            "hard_negative_specificity": round(hard_negative_specificity, 3),
            "boundary_specificity": round(boundary_specificity, 3),
            "nonduplicate_specificity": round(nonduplicate_specificity, 3),
            "overall_accuracy": round(overall_accuracy, 3),
            "false_merges": len(false_merges),
            "missed_duplicates": len(missed_duplicates),
            "PASS_duplicate_recall": duplicate_recall >= 0.80,
            "PASS_hard_negative_specificity": hard_negative_specificity >= 0.85,
            "PASS_boundary_specificity": boundary_specificity >= 0.67,
            "PASS_overall_accuracy": overall_accuracy >= 0.80,
            "PASS": passed,
            "pass_threshold_duplicate_recall": 0.80,
            "pass_threshold_hard_negative_specificity": 0.85,
            "pass_threshold_boundary_specificity": 0.67,
            "pass_threshold_overall_accuracy": 0.80,
        },
        "false_merges": false_merges,
        "missed_duplicates": missed_duplicates,
        "case_results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D1: Duplicate Calibration")
    print("=" * 60)
    print(f"Duplicate recall        : {duplicate_recall:.1%} (threshold: 80%)")
    print(
        "Hard-negative specificity: "
        f"{hard_negative_specificity:.1%} (threshold: 85%)"
    )
    print(f"Boundary specificity    : {boundary_specificity:.1%} (threshold: 67%)")
    print(f"Overall accuracy        : {overall_accuracy:.1%} (threshold: 80%)")
    print(f"False merges            : {len(false_merges)}")
    print(f"Missed duplicates       : {len(missed_duplicates)}")
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
