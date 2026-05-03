"""
Dimension 3 - Test 6: Sub-question Utility
==========================================
Measures whether thinker-generated sub-questions are useful for later research.
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import (
    DEFAULT_MISSION,
    judge_json,
    make_thinker,
    max_restatement_ratio,
    unique_fraction,
)


JUDGE_PROMPT = """
You are evaluating the utility and focus of generated sub-questions.

MAIN QUESTION:
"{question}"

SUB-QUESTIONS (WITH SYSTEM CONTEXT):
{sub_questions}

THINKER'S PROPOSED FOCUS:
"{recommended_focus}"

A useful sub-question should be concrete, non-duplicate, answerable with evidence, and likely to help retrieval or experimentation.
The Thinker has selected one of these questions as the priority focus based on tractability and leverage. 

Score the overall set of sub-questions on three dimensions:
- specificity_score: 1 to 7
- diversity_score: 1 to 7
- downstream_utility_score: 1 to 7

Then, evaluate whether the Thinker's proposed focus is a logically sound and highly useful starting point:
- "focus_is_valid": true or false

Respond EXACTLY in JSON:
{{
  "useful": true or false,
  "specificity_score": 1 to 7,
  "diversity_score": 1 to 7,
  "downstream_utility_score": 1 to 7,
  "focus_is_valid": true or false,
  "reasoning": "one or two sentences explaining your scores and validity judgment"
}}
"""


CASES = [
    "Which sub-questions would best guide research into whether chromatin insulation stabilizes exploratory learning?",
    "How should one decompose the problem of comparing epigenetic memory with policy memory in artificial agents?",
    "What sub-questions would make the analogy between developmental constraint and regularization empirically useful?",
    "How can the exploration-versus-stability problem be broken into researchable questions across biology and machine learning?",
    "What narrower questions would determine when stochastic variation helps rather than harms adaptation?",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim3/results/d3_subquestion_utility.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Evaluating sub-question utility")
    print("=" * 60)

    evaluations = []
    useful_count = 0
    total_score = 0.0
    unique_fraction_total = 0.0
    restatement_ratio_total = 0.0
    focus_validity_count = 0

    for i, question in enumerate(CASES, start=1):
        thinker, _, _, _ = make_thinker(
            mission=DEFAULT_MISSION,
            policy_tag=f"subq_utility_{i}",
        )
        log = thinker.think(question=question, pattern="reductive")
        sub_questions_str = [sq.get("question", "") for sq in log.sub_questions if sq.get("question")]
        uniq_fraction = unique_fraction(sub_questions_str)
        restatement_ratio = max_restatement_ratio(question, sub_questions_str)
        unique_fraction_total += uniq_fraction
        restatement_ratio_total += restatement_ratio

        sub_questions_for_judge = [
            {
                "question": sq.get("question", ""),
                "existing_evidence": sq.get("existing_evidence", "none"),
                "tractability": sq.get("tractability", "medium"),
                "leverage": sq.get("leverage", "medium")
            }
            for sq in log.sub_questions if sq.get("question")
        ]

        recommended_focus = log.insight or ""

        judgment = judge_json(
            JUDGE_PROMPT.format(
                question=question,
                sub_questions=json.dumps(sub_questions_for_judge, indent=2),
                recommended_focus=recommended_focus,
            ),
            model=args.judge_model,
            default={
                "useful": False,
                "specificity_score": 1,
                "diversity_score": 1,
                "downstream_utility_score": 1,
                "focus_is_valid": False,
                "reasoning": "Judge parse failed",
            },
        )
        focus_validity = bool(judgment.get("focus_is_valid", False))
        if focus_validity:
            focus_validity_count += 1
        specificity_score = float(judgment.get("specificity_score", 1))
        diversity_score = float(judgment.get("diversity_score", 1))
        downstream_utility_score = float(judgment.get("downstream_utility_score", 1))
        score = (
            specificity_score
            + diversity_score
            + downstream_utility_score
        ) / 3.0
        useful = bool(judgment.get("useful")) and (
            uniq_fraction >= 0.80 and restatement_ratio <= 0.9
        )
        useful_count += int(useful)
        total_score += score

        evaluations.append(
            {
                "question": question,
                "sub_questions": log.sub_questions,
                "recommended_focus": recommended_focus,
                "unique_fraction": round(uniq_fraction, 3),
                "max_restatement_ratio": round(restatement_ratio, 3),
                "focus_is_valid": focus_validity,
                "judgment": judgment,
            }
        )
        time.sleep(0.2)

    useful_fraction = useful_count / max(len(CASES), 1)
    mean_score = total_score / max(len(CASES), 1)
    mean_unique_fraction = unique_fraction_total / max(len(CASES), 1)
    mean_restatement_ratio = restatement_ratio_total / max(len(CASES), 1)
    focus_validity_fraction = focus_validity_count / max(len(CASES), 1)
    passed = (
        useful_fraction >= 0.60
        and mean_score >= 4.0
        and mean_unique_fraction >= 0.80
        and mean_restatement_ratio <= 0.85
        and focus_validity_fraction >= 0.80
    )

    report = {
        "test": "D3 - Sub-question Utility",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model},
        "summary": {
            "cases_evaluated": len(CASES),
            "useful_count": useful_count,
            "useful_fraction": round(useful_fraction, 3),
            "mean_score": round(mean_score, 3),
            "mean_unique_fraction": round(mean_unique_fraction, 3),
            "mean_max_restatement_ratio": round(mean_restatement_ratio, 3),
            "focus_validity_fraction": round(focus_validity_fraction, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Useful fraction     : {useful_fraction:.2%}")
    print(f"Mean judge score    : {mean_score:.2f}/7")
    print(f"Mean unique fraction: {mean_unique_fraction:.2%}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
