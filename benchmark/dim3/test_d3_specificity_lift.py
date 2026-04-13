"""
Dimension 3 - Test 5: Mission-answer Specificity Lift
=====================================================
Measures whether later rounds of thinking become more specific to the mission
than the initial round.
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

from _shared import add_focus_question, judge_json, make_thinker
from graph.brain import NodeType


JUDGE_PROMPT = """
You are comparing two candidate partial answers to a scientific mission.

MISSION:
"{mission}"

ROUND 1 OUTPUT:
{round1}

FINAL ROUND OUTPUT:
{final_round}

For EACH output, score these dimensions from 1 to 7:
- mission_directness: how directly it answers the mission rather than a broader nearby topic
- mechanistic_specificity: how precisely it names a mechanism, mapping, boundary condition, or structure
- discriminative_value: how much it gives a decisive comparison, test, or decision-relevant distinction

Then choose which output is MORE specific overall.
Avoid ties unless they are genuinely equal.

Respond EXACTLY in JSON:
{{
  "round1_mission_directness": 1 to 7,
  "round1_mechanistic_specificity": 1 to 7,
  "round1_discriminative_value": 1 to 7,
  "final_mission_directness": 1 to 7,
  "final_mechanistic_specificity": 1 to 7,
  "final_discriminative_value": 1 to 7,
  "more_specific": "round1" or "final" or "tie",
  "abstraction_drift": true or false,
  "reasoning": "one or two sentences"
}}
"""


CASES = [
    {
        "mission": "How do biological and artificial learning systems balance exploration with stability?",
        "seed_question": "What mechanism could make exploratory updates safe without freezing adaptation?",
    },
    {
        "mission": "Can epigenetic memory be treated as a protected latent state rather than simple persistence?",
        "seed_question": "What observation would separate true state memory from temporary biochemical inertia?",
    },
    {
        "mission": "When is developmental constraint helpful rather than harmful for adaptation?",
        "seed_question": "What evidence would distinguish constraint as regularization from constraint as mere limitation?",
    },
    {
        "mission": "Can thermodynamic landscape ideas clarify biological exploration-versus-stability trade-offs?",
        "seed_question": "Which mapping would make the landscape analogy specific enough to be useful?",
    },
]


def _best_text(log):
    insight = (log.insight or "").strip()
    if insight:
        if insight.lower().startswith("priority focus:"):
            return insight
        if not re.match(r"^(What|Which|How|Why|When|Where|Can|Could|Is|Are|Under what)\b", insight):
            return insight
    if log.reasoning:
        return log.reasoning
    if insight:
        return f"Priority focus: {insight}"
    return log.question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim3/results/d3_specificity_lift.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Measuring specificity lift across thinking rounds")
    print("=" * 60)

    evaluations = []
    improved_count = 0
    total_lift = 0.0
    abstraction_drift_count = 0
    parse_failed_count = 0

    for i, case in enumerate(CASES, start=1):
        thinker, brain, _, emb_index = make_thinker(
            mission=case["mission"],
            policy_tag=f"specificity_{i}",
        )
        add_focus_question(
            brain,
            emb_index,
            case["seed_question"],
            node_type=NodeType.QUESTION,
            cluster="thinking",
            importance=0.9,
        )
        logs = thinker.think_session(num_rounds=3)
        round1_text = _best_text(logs[0])
        final_text = _best_text(logs[-1])

        judgment = judge_json(
            JUDGE_PROMPT.format(
                mission=case["mission"],
                round1=round1_text,
                final_round=final_text,
            ),
            model=args.judge_model,
            default={
                "round1_mission_directness": 1,
                "round1_mechanistic_specificity": 1,
                "round1_discriminative_value": 1,
                "final_mission_directness": 1,
                "final_mechanistic_specificity": 1,
                "final_discriminative_value": 1,
                "more_specific": "round1",
                "abstraction_drift": False,
                "reasoning": "Judge parse failed",
            },
        )
        round1_score = (
            float(judgment.get("round1_mission_directness", 1))
            + float(judgment.get("round1_mechanistic_specificity", 1))
            + float(judgment.get("round1_discriminative_value", 1))
        ) / 3.0
        final_score = (
            float(judgment.get("final_mission_directness", 1))
            + float(judgment.get("final_mechanistic_specificity", 1))
            + float(judgment.get("final_discriminative_value", 1))
        ) / 3.0
        lift = final_score - round1_score
        improved = judgment.get("more_specific") == "final" or lift > 0.25
        improved_count += int(improved)
        total_lift += lift
        abstraction_drift_count += int(bool(judgment.get("abstraction_drift")))
        parse_failed_count += int(bool(judgment.get("_parse_failed")))

        evaluations.append(
            {
                "mission": case["mission"],
                "seed_question": case["seed_question"],
                "round1_output": round1_text,
                "final_output": final_text,
                "judgment": judgment,
                "round1_score": round(round1_score, 3),
                "final_score": round(final_score, 3),
                "lift": round(lift, 3),
            }
        )
        time.sleep(0.2)

    improved_fraction = improved_count / max(len(CASES), 1)
    mean_lift = total_lift / max(len(CASES), 1)
    abstraction_drift_fraction = abstraction_drift_count / max(len(CASES), 1)
    passed = (
        improved_fraction >= 0.50
        and mean_lift > 0.25
        and abstraction_drift_fraction <= 0.34
    )

    report = {
        "test": "D3 - Mission-answer Specificity Lift",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model},
        "summary": {
            "cases_evaluated": len(CASES),
            "improved_count": improved_count,
            "improved_fraction": round(improved_fraction, 3),
            "mean_lift": round(mean_lift, 3),
            "abstraction_drift_count": abstraction_drift_count,
            "abstraction_drift_fraction": round(abstraction_drift_fraction, 3),
            "parse_failed_count": parse_failed_count,
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Improved fraction : {improved_fraction:.2%}")
    print(f"Mean lift         : {mean_lift:+.2f}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
