"""
Dimension 3 - Test 2: Reductive Sub-question Sufficiency
========================================================
Measures whether Thinker's reductive mode breaks hard questions into useful,
answer-bearing sub-questions.
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

from _shared import (
    DEFAULT_MISSION,
    judge_json,
    make_thinker,
    max_restatement_ratio,
    unique_fraction,
)


JUDGE_PROMPT = """
You are evaluating whether a decomposition of a scientific question is useful.

MAIN QUESTION:
"{question}"

GENERATED SUB-QUESTIONS:
{sub_questions}

RECOMMENDED FOCUS:
{focus}

Score the decomposition on three dimensions:
- coverage_score: does it cover the major unknowns needed to answer the main question?
- non_redundancy_score: are the sub-questions materially distinct rather than restatements?
- answerability_score: are they concrete enough to guide evidence gathering or modeling?

Avoid giving all 7s unless the decomposition is exceptionally strong.

Respond EXACTLY in JSON:
{{
  "sufficient": true or false,
  "coverage_score": 1 to 7,
  "non_redundancy_score": 1 to 7,
  "answerability_score": 1 to 7,
  "reasoning": "one or two sentences"
}}
"""


CASES = [
    "How do chromatin marks preserve stable transcriptional identity while still permitting exploratory adaptation?",
    "What would need to be understood to determine whether developmental constraints act like regularization in artificial learning systems?",
    "How can one decide whether selection pressure and annealing are genuinely analogous optimization processes?",
    "What simpler unknowns would resolve whether epigenetic memory can buffer exploratory noise without erasing useful state?",
    "How should the exploration-versus-stability problem be decomposed if the goal is to compare biological and artificial learners?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim3/results/d3_reductive_sufficiency.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Running reductive thinking cases")
    print("=" * 60)

    evaluations = []
    sufficient_count = 0
    total_score = 0.0
    total_sub_questions = 0
    unique_fraction_total = 0.0
    restatement_ratio_total = 0.0

    for i, question in enumerate(CASES, start=1):
        thinker, _, _, _ = make_thinker(
            mission=DEFAULT_MISSION,
            policy_tag=f"reductive_{i}",
        )
        log = thinker.think(question=question, pattern="reductive")
        sub_question_texts = [
            sq.get("question", "")
            for sq in log.sub_questions
            if sq.get("question")
        ]
        uniq_fraction = unique_fraction(sub_question_texts)
        restatement_ratio = max_restatement_ratio(question, sub_question_texts)

        judgment = judge_json(
            JUDGE_PROMPT.format(
                question=question,
                sub_questions=json.dumps(log.sub_questions, indent=2),
                focus=log.insight or "none",
            ),
            model=args.judge_model,
            default={
                "sufficient": False,
                "coverage_score": 1,
                "non_redundancy_score": 1,
                "answerability_score": 1,
                "reasoning": "Judge parse failed",
            },
        )
        coverage_score = float(judgment.get("coverage_score", 1))
        non_redundancy_score = float(judgment.get("non_redundancy_score", 1))
        answerability_score = float(judgment.get("answerability_score", 1))
        score = (
            coverage_score
            + non_redundancy_score
            + answerability_score
        ) / 3.0
        sufficient = bool(judgment.get("sufficient")) and (
            uniq_fraction >= 0.75 and restatement_ratio <= 0.9
        )
        sufficient_count += int(sufficient)
        total_score += score
        total_sub_questions += len(log.sub_questions)
        unique_fraction_total += uniq_fraction
        restatement_ratio_total += restatement_ratio

        evaluations.append(
            {
                "question": question,
                "sub_questions": log.sub_questions,
                "recommended_focus": log.insight,
                "sub_question_count": len(log.sub_questions),
                "unique_fraction": round(uniq_fraction, 3),
                "max_restatement_ratio": round(restatement_ratio, 3),
                "judgment": judgment,
            }
        )
        time.sleep(0.2)

    sufficient_fraction = sufficient_count / max(len(CASES), 1)
    mean_score = total_score / max(len(CASES), 1)
    avg_sub_questions = total_sub_questions / max(len(CASES), 1)
    mean_unique_fraction = unique_fraction_total / max(len(CASES), 1)
    mean_restatement_ratio = restatement_ratio_total / max(len(CASES), 1)
    passed = (
        sufficient_fraction >= 0.60
        and mean_score >= 4.0
        and avg_sub_questions >= 2.0
        and mean_unique_fraction >= 0.80
        and mean_restatement_ratio <= 0.85
    )

    report = {
        "test": "D3 - Reductive Sub-question Sufficiency",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model},
        "summary": {
            "cases_evaluated": len(CASES),
            "sufficient_count": sufficient_count,
            "sufficient_fraction": round(sufficient_fraction, 3),
            "mean_score": round(mean_score, 3),
            "avg_sub_questions": round(avg_sub_questions, 3),
            "mean_unique_fraction": round(mean_unique_fraction, 3),
            "mean_max_restatement_ratio": round(mean_restatement_ratio, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Sufficient fraction : {sufficient_fraction:.2%}")
    print(f"Mean judge score    : {mean_score:.2f}/7")
    print(f"Avg sub-questions   : {avg_sub_questions:.2f}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
