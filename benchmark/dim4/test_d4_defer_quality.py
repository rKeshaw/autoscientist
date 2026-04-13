"""
Dimension 4 - Test 5: Defer Quality
====================================
Tests whether the Critic's DEFER verdicts are appropriate — i.e., the
deferred claims are legitimately uncertain (not secretly good claims
that should have been accepted, or obviously bad claims that should
have been rejected).

HYBRID APPROACH: Claims are handcrafted to be genuinely ambiguous, but
context is NOT injected — the Critic builds its own from the real graph.

NOTE: Non-DEFER verdicts are still reported for diagnosis, but the
benchmark only passes if the Critic actually exercises the DEFER path
and those deferrals are judged appropriate.

Benchmark level:
  - module-level
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
from critic.critic import CandidateThought, Verdict


DEFER_JUDGE_PROMPT = """
You are evaluating whether a Critic's decision to DEFER a scientific claim
was appropriate.

CLAIM: "{claim}"
VERDICT ISSUED: defer
CONFIDENCE: {confidence}
VERDICT REASON: "{verdict_reason}"

A DEFER verdict means the Critic decided there wasn't enough evidence to
accept or reject the claim. Evaluate whether this was a good decision:

- "too_good_to_defer": Was this claim obviously strong enough to accept?
- "too_bad_to_defer": Was this claim obviously flawed enough to reject?
- "appropriately_uncertain": Was deferral reasonable given the evidence?

Score:
- defer_quality: 1 to 7 (was deferring the right call?)

Respond EXACTLY in JSON:
{{
  "too_good_to_defer": true or false,
  "too_bad_to_defer": true or false,
  "appropriately_uncertain": true or false,
  "defer_quality": 1 to 7,
  "reasoning": "one or two sentences explaining your assessment"
}}
"""

NON_DEFER_JUDGE_PROMPT = """
You are evaluating the Critic's verdict on a genuinely uncertain scientific claim.
The claim was designed to be ambiguous — plausible but underspecified.

CLAIM: "{claim}"
VERDICT ISSUED: "{verdict}"
CONFIDENCE: {confidence}
VERDICT REASON: "{verdict_reason}"
FINAL CLAIM: "{final_claim}"

Was the Critic's verdict reasonable for this genuinely uncertain claim?
The claim is in a gray zone — accept, refine, or defer could all be defensible,
but outright reject (with high confidence) would be surprising.

Score:
- verdict_quality: 1 to 7 (was the verdict reasonable for an uncertain claim?)

Respond EXACTLY in JSON:
{{
  "verdict_reasonable": true or false,
  "verdict_quality": 1 to 7,
  "reasoning": "one or two sentences explaining your assessment"
}}
"""


# ── Claims in the uncertain middle zone ──────────────────────────────────────
# No context provided — Critic builds its own from graph.

CASES = [
    (
        "The topological structure of chromatin territories during interphase "
        "may encode constraints on evolutionary search that are functionally "
        "equivalent to priors in Bayesian optimization.",
        "Plausible topology-to-prior mapping, lacking formal grounding",
    ),
    (
        "The error-correcting properties of the genetic code may be "
        "analogous to the error-correcting properties of convolutional "
        "codes in information theory, suggesting a deeper connection "
        "between biological information storage and engineered "
        "communication systems.",
        "Error-correction analogy — partially explored, not conclusive",
    ),
    (
        "Neuromodulatory systems in the brain implement a form of "
        "meta-learning that is structurally equivalent to hyperparameter "
        "optimization in neural architecture search.",
        "Meta-learning analogy — intriguing but mechanism unclear",
    ),
    (
        "The balance between synaptic potentiation and depression in "
        "biological neural circuits implements a natural form of gradient "
        "clipping that prevents catastrophic weight updates.",
        "Gradient clipping analogy — suggestive but mechanistically distinct",
    ),
    (
        "Horizontal gene transfer between bacterial species implements "
        "a form of knowledge distillation where successfully evolved "
        "solutions are compressed and transferred to naive learners.",
        "Knowledge distillation analogy — interesting but speculative",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim4/results/d4_defer_quality.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 5: Defer Quality (graph context, no escape hatch)")
    print("=" * 60)

    critic, _, _, _ = make_critic(with_insight_buffer=True)

    evaluations = []
    deferred_count = 0
    appropriate_defers = 0
    total_defer_quality = 0.0
    non_defer_reasonable = 0
    total_non_defer_quality = 0.0

    for claim, description in CASES:
        print(f"\n  Case: {description}")
        # Context is intentionally empty — Critic builds its own from graph
        candidate = CandidateThought(
            claim=claim,
            source_module="thinker",
            proposed_type="synthesis",
            importance=0.75,
            context="",   # <-- hybrid approach: no injected context
        )

        log = critic.evaluate_with_refinement(candidate)

        if log.verdict == Verdict.DEFER:
            deferred_count += 1

            judgment = judge_json(
                DEFER_JUDGE_PROMPT.format(
                    claim=claim,
                    confidence=log.confidence,
                    verdict_reason=log.verdict_reason,
                ),
                model=args.judge_model,
                default={
                    "too_good_to_defer": False,
                    "too_bad_to_defer": False,
                    "appropriately_uncertain": True,
                    "defer_quality": 4,
                    "reasoning": "Judge parse failed",
                },
            )

            is_appropriate = bool(judgment.get("appropriately_uncertain", False))
            if is_appropriate:
                appropriate_defers += 1
            defer_quality = float(judgment.get("defer_quality", 1))
            total_defer_quality += defer_quality

            evaluations.append({
                "description": description,
                "claim": claim,
                "verdict": log.verdict.value,
                "confidence": log.confidence,
                "verdict_reason": log.verdict_reason,
                "appropriately_uncertain": is_appropriate,
                "defer_quality": defer_quality,
                "judgment": judgment,
            })

            status = "✓" if is_appropriate else "✗"
            print(f"  {status} DEFERRED (quality={defer_quality:.0f}/7) — "
                  f"{judgment.get('reasoning', '')[:80]}")
        else:
            # Judge the non-defer verdict for reasonableness
            judgment = judge_json(
                NON_DEFER_JUDGE_PROMPT.format(
                    claim=claim,
                    verdict=log.verdict.value,
                    confidence=log.confidence,
                    verdict_reason=log.verdict_reason,
                    final_claim=log.final_claim or claim,
                ),
                model=args.judge_model,
                default={
                    "verdict_reasonable": False,
                    "verdict_quality": 4,
                    "reasoning": "Judge parse failed",
                },
            )

            is_reasonable = bool(judgment.get("verdict_reasonable", False))
            if is_reasonable:
                non_defer_reasonable += 1
            quality = float(judgment.get("verdict_quality", 1))
            total_non_defer_quality += quality

            evaluations.append({
                "description": description,
                "claim": claim,
                "verdict": log.verdict.value,
                "confidence": log.confidence,
                "verdict_reason": log.verdict_reason,
                "final_claim": log.final_claim,
                "verdict_reasonable": is_reasonable,
                "verdict_quality": quality,
                "judgment": judgment,
            })

            status = "✓" if is_reasonable else "✗"
            print(f"  {status} {log.verdict.value.upper()} (quality={quality:.0f}/7"
                  f", conf={log.confidence:.2f}) — "
                  f"{judgment.get('reasoning', '')[:80]}")

        time.sleep(0.2)

    n = max(len(CASES), 1)
    defer_fraction = deferred_count / n
    non_defer_count = n - deferred_count

    appropriate_defer_fraction = (
        appropriate_defers / max(deferred_count, 1) if deferred_count > 0 else 0.0
    )
    mean_defer_quality = (
        total_defer_quality / max(deferred_count, 1) if deferred_count > 0 else 0.0
    )
    non_defer_reasonable_fraction = (
        non_defer_reasonable / max(non_defer_count, 1) if non_defer_count > 0 else 0.0
    )
    mean_non_defer_quality = (
        total_non_defer_quality / max(non_defer_count, 1) if non_defer_count > 0 else 0.0
    )

    passed = (
        deferred_count > 0 and
        appropriate_defer_fraction >= 0.60 and
        mean_defer_quality >= 4.0
    )

    report = {
        "test": "D4 - Defer Quality",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model, "context_source": "graph"},
        "summary": {
            "cases_evaluated": len(CASES),
            "deferred_count": deferred_count,
            "defer_fraction": round(defer_fraction, 3),
            "appropriate_defer_fraction": round(appropriate_defer_fraction, 3),
            "mean_defer_quality": round(mean_defer_quality, 3),
            "non_defer_count": non_defer_count,
            "non_defer_reasonable_fraction": round(non_defer_reasonable_fraction, 3),
            "mean_non_defer_quality": round(mean_non_defer_quality, 3),
            "PASS_requires_actual_defers": deferred_count > 0,
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nDeferred              : {deferred_count}/{len(CASES)}")
    if deferred_count > 0:
        print(f"Appropriate defer rate: {appropriate_defer_fraction:.2%}")
        print(f"Mean defer quality    : {mean_defer_quality:.2f}/7")
    else:
        print("Appropriate defer rate: N/A (no DEFER verdicts were issued)")
    if non_defer_count > 0:
        print(f"Non-defer reasonable  : {non_defer_reasonable_fraction:.2%}")
        print(f"Mean non-defer quality: {mean_non_defer_quality:.2f}/7")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
