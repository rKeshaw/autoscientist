"""
Dimension 4 - Test 2: Verdict Accuracy
=======================================
Tests whether the Critic issues reasonable verdicts (accept/refine/reject/defer)
on claims of known quality.

Uses LENIENT matching: for each test case, an "acceptable_verdicts" set is
defined. The Critic's actual verdict must fall within that set. An LLM judge
then independently evaluates whether each verdict was reasonable ON ITS OWN
MERITS — the judge does NOT see our expected category to avoid circular bias.

HYBRID APPROACH: Claims are handcrafted to control difficulty, but context
is NOT injected — the Critic builds its own context from the real graph
via _build_context(), testing the full end-to-end pipeline.
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


# The judge evaluates the verdict on its own merits — no expected_category
# is provided to avoid circular bias where the judge just penalizes
# disagreement with our pre-determined labels.
JUDGE_PROMPT = """
You are evaluating whether a Critic's verdict on a scientific claim was reasonable.

IMPORTANT CONTEXT: The Critic's role is to filter claims for NON-TRIVIAL
contributions to a scientific knowledge graph. A factually correct claim
can still be reasonably REJECTED if it merely restates common textbook
knowledge without adding novel insight (e.g., "water is H2O" is true but
trivially obvious and adds no scientific value to a knowledge graph).

CLAIM: "{claim}"
VERDICT ISSUED: "{verdict}"
VERDICT REASON: "{verdict_reason}"
CONFIDENCE: {confidence}
FINAL CLAIM (after dialogue): "{final_claim}"

Was the Critic's verdict reasonable? Consider:
1. Is the verdict defensible given the claim's scientific quality AND novelty?
2. Is the confidence score well-calibrated for the verdict?
3. Is the reasoning sound and specific?

Score on three dimensions:
- verdict_reasonableness: 1 to 7 (was the verdict defensible?)
- confidence_calibration: 1 to 7 (does the confidence appropriately reflect certainty?)
- reasoning_quality: 1 to 7 (is the explanation sound and specific?)

Respond EXACTLY in JSON:
{{
  "verdict_reasonable": true or false,
  "verdict_reasonableness": 1 to 7,
  "confidence_calibration": 1 to 7,
  "reasoning_quality": 1 to 7,
  "reasoning": "one or two sentences explaining your assessment"
}}
"""


# ── Test cases ───────────────────────────────────────────────────────────────
# Acceptable verdicts are tightened:
# - should_accept → {accept} only (strong claims must be accepted)
# - should_reject → {reject} only (bad claims must be rejected)
# - genuinely ambiguous → broader set
# NO pre-written context — the Critic builds its own from the graph.

CASES = [
    # Should-accept: well-grounded synthesis
    (
        "The balance between exploration and exploitation in reinforcement "
        "learning agents mirrors the biological tension between genetic "
        "variation and stabilizing selection, with both systems requiring "
        "mechanisms to prevent catastrophic forgetting of validated strategies.",
        "should_accept",
        {Verdict.ACCEPT},
        "Well-grounded cross-domain synthesis",
    ),
    # Should-accept: clear structural analogy
    (
        "Synaptic consolidation during sleep functions as a biological "
        "analogue of experience replay in deep reinforcement learning, "
        "where offline reprocessing of episodic memories stabilizes learned "
        "policies against catastrophic interference.",
        "should_accept",
        {Verdict.ACCEPT},
        "Well-supported structural analogy with explicit mapping",
    ),
    # Should-reject: incoherent claim
    (
        "DNA methylation causes the entropy of the universe to decrease "
        "locally, which proves that mutation is thermodynamically impossible "
        "in closed systems.",
        "should_reject",
        {Verdict.REJECT},
        "Scientifically incoherent — misapplies thermodynamics",
    ),
    # Should-reject: trivially obvious
    (
        "Organisms that are better adapted to their environment tend to "
        "survive and reproduce more successfully.",
        "should_reject",
        {Verdict.REJECT},
        "Trivially restates the definition of natural selection",
    ),
    # Genuinely ambiguous: plausible but under-evidenced
    (
        "The topological structure of gene regulatory networks encodes a "
        "form of implicit memory that constrains future evolutionary "
        "trajectories in ways analogous to how network architecture "
        "constrains learning capacity in artificial neural networks.",
        "ambiguous",
        {Verdict.ACCEPT, Verdict.REFINE, Verdict.DEFER},
        "Plausible structural analogy — genuinely ambiguous",
    ),
    # Should-refine: kernel of truth but poorly formulated
    (
        "Evolution is basically the same as gradient descent because "
        "both find optimal solutions.",
        "should_refine",
        {Verdict.REFINE, Verdict.REJECT},
        "Oversimplified analogy that needs qualification",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim4/results/d4_verdict_accuracy.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 2: Verdict Accuracy (tight matching, unbiased judge)")
    print("=" * 60)

    critic, _, _, _ = make_critic()

    evaluations = []
    correct_count = 0
    total_score = 0.0
    total_confidence_score = 0.0

    for claim, expected_category, acceptable_verdicts, description in CASES:
        print(f"\n  Case: {description}")
        candidate = CandidateThought(
            claim=claim,
            source_module="thinker",
            proposed_type="synthesis",
            importance=0.8,
            context="",   # hybrid approach: no injected context
        )

        log = critic.evaluate(candidate)
        verdict_correct = log.verdict in acceptable_verdicts

        if verdict_correct:
            correct_count += 1

        # Judge the verdict — NO expected_category to avoid circular bias
        judgment = judge_json(
            JUDGE_PROMPT.format(
                claim=claim,
                verdict=log.verdict.value,
                verdict_reason=log.verdict_reason,
                confidence=log.confidence,
                final_claim=log.final_claim or claim,
            ),
            model=args.judge_model,
            default={
                "verdict_reasonable": False,
                "verdict_reasonableness": 1,
                "confidence_calibration": 1,
                "reasoning_quality": 1,
                "reasoning": "Judge parse failed",
            },
        )

        reasonableness = float(judgment.get("verdict_reasonableness", 1))
        calibration = float(judgment.get("confidence_calibration", 1))
        reasoning = float(judgment.get("reasoning_quality", 1))
        score = (reasonableness + calibration + reasoning) / 3.0
        total_score += score
        total_confidence_score += calibration

        evaluations.append({
            "description": description,
            "claim": claim,
            "expected_category": expected_category,
            "acceptable_verdicts": [v.value for v in acceptable_verdicts],
            "actual_verdict": log.verdict.value,
            "verdict_correct": verdict_correct,
            "confidence": log.confidence,
            "verdict_reason": log.verdict_reason,
            "rejection_reason": log.rejection_reason,
            "final_claim": log.final_claim,
            "dialogue_turns": len(log.dialogue),
            "judgment": judgment,
        })

        status = "✓" if verdict_correct else "✗"
        print(f"  {status} Expected: {[v.value for v in acceptable_verdicts]}, "
              f"Got: {log.verdict.value} (conf={log.confidence:.2f})")
        print(f"    Judge score: {score:.1f}/7 — {judgment.get('reasoning', '')[:80]}")
        time.sleep(0.2)

    n = max(len(CASES), 1)
    verdict_accuracy = correct_count / n
    mean_score = total_score / n
    mean_confidence_calibration = total_confidence_score / n

    passed = verdict_accuracy >= 0.60 and mean_score >= 4.5

    report = {
        "test": "D4 - Verdict Accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model, "matching": "tight",
                   "context_source": "graph", "judge_bias": "unbiased"},
        "summary": {
            "cases_evaluated": len(CASES),
            "correct_count": correct_count,
            "verdict_accuracy": round(verdict_accuracy, 3),
            "mean_judge_score": round(mean_score, 3),
            "mean_confidence_calibration": round(mean_confidence_calibration, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nVerdict accuracy      : {verdict_accuracy:.2%}")
    print(f"Mean judge score      : {mean_score:.2f}/7")
    print(f"Confidence calibration: {mean_confidence_calibration:.2f}/7")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
