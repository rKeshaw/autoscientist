"""
Dimension 4 - Test 4: Refinement Quality Lift
==============================================
Tests whether the Critic's evaluate_with_refinement() loop produces claims
that are measurably better than the input. Feeds deliberately weak-but-
improvable claims and uses an LLM judge to score before vs after.

HYBRID APPROACH: Claims are handcrafted to be deliberately weak, but
context is NOT injected — the Critic builds its own from the real graph.
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

from _shared import make_critic, judge_json
from critic.critic import CandidateThought


JUDGE_PROMPT = """
You are comparing two versions of a scientific claim: the original and a
refined version produced after adversarial review.

ORIGINAL CLAIM:
"{original_claim}"

REFINED CLAIM:
"{final_claim}"

VERDICT: {verdict}
CONFIDENCE: {confidence}

Score EACH version on three dimensions (1 to 7):
- specificity: How precise and non-vague is the claim?
- defensibility: How well could the claim withstand scientific scrutiny?
- informativeness: Does the claim say something non-trivial and useful?

Then determine which version is better.

Respond EXACTLY in JSON:
{{
  "original_specificity": 1 to 7,
  "original_defensibility": 1 to 7,
  "original_informativeness": 1 to 7,
  "final_specificity": 1 to 7,
  "final_defensibility": 1 to 7,
  "final_informativeness": 1 to 7,
  "improved": true or false,
  "degraded": true or false,
  "reasoning": "one or two sentences explaining the comparison"
}}
"""


# ── Weak-but-improvable claims ───────────────────────────────────────────────
# No context provided — the Critic will build its own from the graph.

CASES = [
    (
        "Evolution is like machine learning because both optimize things.",
        "Oversimplified analogy that needs mechanistic precision",
    ),
    (
        "The brain learns the same way neural networks do.",
        "False equivalence that needs qualification",
    ),
    (
        "Gene regulation is basically a computer program.",
        "Misleading computational metaphor",
    ),
    (
        "Epigenetics proves that Lamarck was right about inheritance.",
        "Overstated historical claim",
    ),
    (
        "Information theory explains everything about biology.",
        "Scope overreach with a kernel of truth",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim4/results/d4_refinement_lift.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 4: Refinement Quality Lift (graph context)")
    print("=" * 60)

    critic, _, _, _ = make_critic()

    evaluations = []
    improved_count = 0
    degraded_count = 0
    total_lift = 0.0

    for original_claim, description in CASES:
        print(f"\n  Case: {description}")
        # Context is intentionally empty — Critic builds its own from graph
        candidate = CandidateThought(
            claim=original_claim,
            source_module="thinker",
            proposed_type="synthesis",
            importance=0.8,
            context="",   # <-- hybrid approach: no injected context
        )

        log = critic.evaluate_with_refinement(candidate)

        # Judge before vs after
        judgment = judge_json(
            JUDGE_PROMPT.format(
                original_claim=original_claim,
                final_claim=log.final_claim or original_claim,
                verdict=log.verdict.value,
                confidence=log.confidence,
            ),
            model=args.judge_model,
            default={
                "original_specificity": 1,
                "original_defensibility": 1,
                "original_informativeness": 1,
                "final_specificity": 1,
                "final_defensibility": 1,
                "final_informativeness": 1,
                "improved": False,
                "degraded": False,
                "reasoning": "Judge parse failed",
            },
        )

        orig_score = (
            float(judgment.get("original_specificity", 1))
            + float(judgment.get("original_defensibility", 1))
            + float(judgment.get("original_informativeness", 1))
        ) / 3.0
        final_score = (
            float(judgment.get("final_specificity", 1))
            + float(judgment.get("final_defensibility", 1))
            + float(judgment.get("final_informativeness", 1))
        ) / 3.0
        lift = final_score - orig_score

        improved = bool(judgment.get("improved", False)) or lift > 0.0
        degraded = bool(judgment.get("degraded", False)) and lift < 0.0

        if improved:
            improved_count += 1
        if degraded:
            degraded_count += 1
        total_lift += lift

        evaluations.append({
            "description": description,
            "original_claim": original_claim,
            "final_claim": log.final_claim or original_claim,
            "verdict": log.verdict.value,
            "confidence": log.confidence,
            "original_score": round(orig_score, 3),
            "final_score": round(final_score, 3),
            "lift": round(lift, 3),
            "improved": improved,
            "degraded": degraded,
            "judgment": judgment,
        })

        status = "↑" if improved else ("↓" if degraded else "→")
        print(f"  {status} Original: {orig_score:.1f} → Final: {final_score:.1f} "
              f"(lift={lift:+.1f}, verdict={log.verdict.value})")
        print(f"    {judgment.get('reasoning', '')[:80]}")
        time.sleep(0.2)

    n = max(len(CASES), 1)
    improved_fraction = improved_count / n
    degraded_fraction = degraded_count / n
    mean_lift = total_lift / n

    passed = (
        improved_fraction >= 0.60
        and mean_lift > 0.0
        and degraded_fraction <= 0.20
    )

    report = {
        "test": "D4 - Refinement Quality Lift",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model, "context_source": "graph"},
        "summary": {
            "cases_evaluated": len(CASES),
            "improved_count": improved_count,
            "improved_fraction": round(improved_fraction, 3),
            "degraded_count": degraded_count,
            "degraded_fraction": round(degraded_fraction, 3),
            "mean_lift": round(mean_lift, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nImproved fraction : {improved_fraction:.2%}")
    print(f"Degraded fraction : {degraded_fraction:.2%}")
    print(f"Mean lift         : {mean_lift:+.3f}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
