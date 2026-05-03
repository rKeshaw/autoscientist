"""
Dimension 3 - Test 3: Insight Actionability
===========================================
Measures whether Thinker's deliberate reasoning produces next-step-useful
insights instead of generic prose.
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

from _shared import DEFAULT_MISSION, judge_json, make_thinker


JUDGE_PROMPT = """
You are evaluating whether a reasoning output is actionable for scientific work.

MISSION:
"{mission}"

QUESTION:
"{question}"

PATTERN:
{pattern}

REASONING:
{reasoning}

EXTRACTED INSIGHT:
{insight}

Score the output on three dimensions:
- mission_directness: how directly it advances the mission
- next_step_concreteness: how clearly it suggests a concrete next step, experiment, or comparison
- discriminative_value: how much it distinguishes one explanation, prediction, or decision from another

An actionable output should score well on most dimensions. Avoid giving all 7s unless the output is exceptional.

Respond EXACTLY in JSON:
{{
  "actionable": true or false,
  "mission_directness": 1 to 7,
  "next_step_concreteness": 1 to 7,
  "discriminative_value": 1 to 7,
  "reasoning": "one or two sentences"
}}
"""


CASES = [
    {
        "question": "What evidence supports versus weakens the claim that epigenetic gating functions like protected memory in adaptive learning?",
        "pattern": "dialectical",
    },
    {
        "question": "What analogous control problem from another domain could help explain stability-preserving exploration in developmental systems?",
        "pattern": "analogical",
    },
    {
        "question": "If exploratory noise is useful only when state memory is partially insulated, what observation should distinguish that hypothesis from pure instability?",
        "pattern": "experimental",
    },
    {
        "question": "What unifying principle could explain why chromatin remodeling, annealing, and selection all appear to regulate constrained search?",
        "pattern": "integrative",
    },
    {
        "question": "What evidence should be weighed for and against the idea that fixed developmental constraints play the same role as regularizers in learning systems?",
        "pattern": "dialectical",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim3/results/d3_actionability.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Running thinker actionability cases")
    print("=" * 60)

    evaluations = []
    actionable_count = 0
    total_score = 0.0
    parse_failed_count = 0

    for i, case in enumerate(CASES, start=1):
        thinker, _, _, _ = make_thinker(
            mission=DEFAULT_MISSION,
            policy_tag=f"actionability_{i}",
        )
        log = thinker.think(question=case["question"], pattern=case["pattern"])
        judgment = judge_json(
            JUDGE_PROMPT.format(
                mission=DEFAULT_MISSION,
                question=case["question"],
                pattern=case["pattern"],
                reasoning=log.reasoning,
                insight=log.insight or "none",
            ),
            model=args.judge_model,
            default={
                "actionable": False,
                "mission_directness": 1,
                "next_step_concreteness": 1,
                "discriminative_value": 1,
                "reasoning": "Judge parse failed",
            },
        )
        mission_directness = float(judgment.get("mission_directness", 1))
        next_step_concreteness = float(judgment.get("next_step_concreteness", 1))
        discriminative_value = float(judgment.get("discriminative_value", 1))
        score = (
            mission_directness
            + next_step_concreteness
            + discriminative_value
        ) / 3.0
        parse_failed_count += int(bool(judgment.get("_parse_failed")))
        actionable = bool(judgment.get("actionable")) or (
            score >= 4.5
            and next_step_concreteness >= 4
            and discriminative_value >= 4
        )
        actionable_count += int(actionable)
        total_score += score

        evaluations.append(
            {
                "question": case["question"],
                "pattern": case["pattern"],
                "reasoning": log.reasoning,
                "insight": log.insight,
                "score": round(score, 3),
                "judgment": judgment,
            }
        )
        time.sleep(0.2)

    actionable_fraction = actionable_count / max(len(CASES), 1)
    mean_score = total_score / max(len(CASES), 1)
    passed = actionable_fraction >= 0.60 and mean_score >= 4.0

    report = {
        "test": "D3 - Insight Actionability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model},
        "summary": {
            "cases_evaluated": len(CASES),
            "actionable_count": actionable_count,
            "actionable_fraction": round(actionable_fraction, 3),
            "mean_score": round(mean_score, 3),
            "parse_failed_count": parse_failed_count,
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Actionable fraction : {actionable_fraction:.2%}")
    print(f"Mean judge score    : {mean_score:.2f}/7")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
