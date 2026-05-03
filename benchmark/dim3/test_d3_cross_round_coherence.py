"""
Dimension 3 - Test 4: Cross-round Coherence
===========================================
Measures whether multi-round thinking sessions stay on-topic and build on prior
rounds rather than drifting arbitrarily.
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

from _shared import add_focus_question, judge_json, make_thinker
from graph.brain import NodeType


JUDGE_PROMPT = """
You are evaluating the coherence of a multi-round scientific thinking session.

MISSION:
"{mission}"

ROUNDS:
{rounds}

Judge whether later rounds:
1. remain tied to the mission,
2. build on earlier reasoning rather than resetting,
3. add useful precision or testing leverage rather than merely becoming more abstract.

Respond EXACTLY in JSON:
{{
  "coherent": true or false,
  "progressive": true or false,
  "abstraction_drift": true or false,
  "coherence_score": 1 to 7,
  "progress_score": 1 to 7,
  "reasoning": "one or two sentences"
}}
"""


SESSIONS = [
    {
        "mission": "How do biological and artificial learning systems balance exploration with stability?",
        "seed_question": "What mechanism could preserve a stable memory core while still permitting exploratory updates?",
    },
    {
        "mission": "Can developmental constraints be understood as a form of regularization on biological search?",
        "seed_question": "What kind of evidence would distinguish true regularization from simple physical limitation?",
    },
    {
        "mission": "When does stochastic variation improve adaptation rather than destabilize it?",
        "seed_question": "Which observations would connect beneficial stochasticity in evolution and machine learning?",
    },
    {
        "mission": "How should epigenetic memory be compared to policy memory in artificial agents?",
        "seed_question": "What structural features are shared, and what would count as a false analogy?",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim3/results/d3_cross_round_coherence.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Running multi-round thinking sessions")
    print("=" * 60)

    evaluations = []
    coherent_count = 0
    progressive_count = 0
    total_score = 0.0
    pattern_diversity_total = 0.0

    for i, session in enumerate(SESSIONS, start=1):
        thinker, brain, _, emb_index = make_thinker(
            mission=session["mission"],
            policy_tag=f"coherence_{i}",
        )
        add_focus_question(
            brain,
            emb_index,
            session["seed_question"],
            node_type=NodeType.QUESTION,
            cluster="thinking",
            importance=0.9,
        )
        logs = thinker.think_session(num_rounds=3)
        rounds_payload = [
            {
                "round": j + 1,
                "question": log.question,
                "pattern": log.pattern,
                "insight": log.insight,
                "sub_questions": log.sub_questions,
            }
            for j, log in enumerate(logs)
        ]
        judgment = judge_json(
            JUDGE_PROMPT.format(
                mission=session["mission"],
                rounds=json.dumps(rounds_payload, indent=2),
            ),
            model=args.judge_model,
            default={
                "coherent": False,
                "progressive": False,
                "abstraction_drift": False,
                "coherence_score": 1,
                "progress_score": 1,
                "reasoning": "Judge parse failed",
            },
        )
        coherent = bool(judgment.get("coherent"))
        progressive = bool(judgment.get("progressive"))
        coherence_score = float(judgment.get("coherence_score", 1))
        progress_score = float(judgment.get("progress_score", 1))
        score = (coherence_score + progress_score) / 2.0
        coherent_count += int(coherent)
        progressive_count += int(progressive)
        total_score += score
        unique_pattern_count = len({item["pattern"] for item in rounds_payload if item.get("pattern")})
        pattern_diversity = unique_pattern_count / max(len(rounds_payload), 1)
        pattern_diversity_total += pattern_diversity

        evaluations.append(
            {
                "mission": session["mission"],
                "seed_question": session["seed_question"],
                "rounds": rounds_payload,
                "unique_pattern_count": unique_pattern_count,
                "pattern_diversity": round(pattern_diversity, 3),
                "judgment": judgment,
            }
        )
        time.sleep(0.2)

    coherent_fraction = coherent_count / max(len(SESSIONS), 1)
    progressive_fraction = progressive_count / max(len(SESSIONS), 1)
    mean_score = total_score / max(len(SESSIONS), 1)
    mean_pattern_diversity = pattern_diversity_total / max(len(SESSIONS), 1)
    passed = (
        coherent_fraction >= 0.67
        and progressive_fraction >= 0.50
        and mean_score >= 4.0
        and mean_pattern_diversity >= 0.50
    )

    report = {
        "test": "D3 - Cross-round Coherence",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model},
        "summary": {
            "sessions_evaluated": len(SESSIONS),
            "coherent_count": coherent_count,
            "coherent_fraction": round(coherent_fraction, 3),
            "progressive_count": progressive_count,
            "progressive_fraction": round(progressive_fraction, 3),
            "mean_score": round(mean_score, 3),
            "mean_pattern_diversity": round(mean_pattern_diversity, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Coherent fraction : {coherent_fraction:.2%}")
    print(f"Progressive frac. : {progressive_fraction:.2%}")
    print(f"Mean judge score  : {mean_score:.2f}/7")
    print(f"Pattern diversity : {mean_pattern_diversity:.2f}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
