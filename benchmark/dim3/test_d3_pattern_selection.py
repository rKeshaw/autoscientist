"""
Dimension 3 - Test 1: Pattern Selection Accuracy
================================================
Measures whether Thinker chooses a plausible reasoning pattern for the question
type it is facing.
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

from _shared import make_thinker, SUPPORTED_PATTERNS
from graph.brain import Node, NodeType, NodeStatus
from embedding import embed as shared_embed


CASES = [
    {
        "question": "Which simpler unknowns must be answered to determine how epigenetic memory preserves stability without preventing adaptation?",
        "node_type": NodeType.QUESTION,
        "cluster": "epigenetics",
        "expected_pattern": "reductive",
        "why": "Complex multi-part question needing decomposition.",
    },
    {
        "question": "What evidence supports the claim that transient chromatin opening can improve exploratory adaptation, and what evidence argues against it?",
        "node_type": NodeType.HYPOTHESIS,
        "cluster": "epigenetics",
        "expected_pattern": "dialectical",
        "why": "Explicitly asks for support and opposition.",
    },
    {
        "question": "Can protein-folding landscapes offer an analogy for balancing stability and exploration in reinforcement learning?",
        "node_type": NodeType.QUESTION,
        "cluster": "thermodynamics",
        "expected_pattern": "analogical",
        "why": "Looks for transfer from another domain.",
    },
    {
        "question": "If exploratory noise truly prevents developmental lock-in, what observable consequence should appear in lineage outcomes?",
        "node_type": NodeType.HYPOTHESIS,
        "cluster": "development",
        "expected_pattern": "experimental",
        "why": "Asks for consequences under true vs false scenarios.",
    },
    {
        "question": "What unifying principle explains why selection, annealing, and chromatin remodeling all appear to regulate constrained search?",
        "node_type": NodeType.GAP,
        "cluster": "thinking",
        "expected_pattern": "integrative",
        "why": "Seeks one principle across multiple facts.",
    },
    {
        "question": "What are the tractable sub-problems behind the question of whether learning systems need a protected memory core?",
        "node_type": NodeType.GAP,
        "cluster": "ann",
        "expected_pattern": "reductive",
        "why": "Asks for sub-problems and leverage.",
    },
    {
        "question": "Is the claim that correlated equilibrium predicts adaptive stability actually defensible, or does the evidence cut the other way?",
        "node_type": NodeType.HYPOTHESIS,
        "cluster": "game_theory",
        "expected_pattern": "dialectical",
        "why": "Contested claim with pro/con structure.",
    },
    {
        "question": "Could the immune system's use of stochastic repertoire generation transfer as a solution template for exploration control in artificial learning?",
        "node_type": NodeType.QUESTION,
        "cluster": "biology",
        "expected_pattern": "analogical",
        "why": "Cross-domain transfer question.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim3/results/d3_pattern_selection.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    thinker, brain, _, emb_index = make_thinker(policy_tag="pattern_selection")

    print("=" * 60)
    print("PHASE 1: Evaluating Thinker pattern selection")
    print("=" * 60)

    evaluations = []
    correct = 0
    unsupported = 0

    for case in CASES:
        node = Node(
            statement=case["question"],
            node_type=case["node_type"],
            cluster=case["cluster"],
            status=NodeStatus.UNCERTAIN,
            importance=0.75,
            source_quality=0.7,
        )
        nid = brain.add_node(node)
        emb_index.add(nid, shared_embed(case["question"]))

        node_type, cluster, chosen = thinker._pick_pattern(case["question"])
        is_supported = chosen in SUPPORTED_PATTERNS
        is_correct = chosen == case["expected_pattern"]
        correct += int(is_correct)
        unsupported += int(not is_supported)

        evaluations.append(
            {
                "question": case["question"],
                "expected_pattern": case["expected_pattern"],
                "chosen_pattern": chosen,
                "matched_node_type": node_type,
                "matched_cluster": cluster,
                "is_supported_pattern": is_supported,
                "is_correct": is_correct,
                "why_expected": case["why"],
            }
        )

    accuracy = correct / max(len(CASES), 1)
    unsupported_rate = unsupported / max(len(CASES), 1)
    passed = accuracy >= 0.60 and unsupported_rate == 0.0

    report = {
        "test": "D3 - Pattern Selection Accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "cases_evaluated": len(CASES),
            "correct_count": correct,
            "accuracy": round(accuracy, 3),
            "unsupported_pattern_rate": round(unsupported_rate, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Accuracy                 : {accuracy:.2%}")
    print(f"Unsupported pattern rate : {unsupported_rate:.2%}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
