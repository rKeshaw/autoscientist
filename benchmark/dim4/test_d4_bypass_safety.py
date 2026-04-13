"""
Dimension 4 - Test 6: Bypass Safety
====================================
Tests the Critic's laziness principle: does bypassing low-stakes claims
correctly avoid letting problematic high-stakes claims through?

This test is entirely deterministic — no LLM calls.
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

from _shared import make_critic
from critic.critic import CandidateThought, Verdict


# ── Test cases ───────────────────────────────────────────────────────────────
# Each: (candidate_kwargs, should_bypass, is_safe_to_bypass, description)
#
# "should_bypass" = True if needs_review() should return False (i.e. bypass)
# "is_safe_to_bypass" = True if bypassing this claim is genuinely safe
# A failure is when should_bypass=True but is_safe_to_bypass=False

CASES = [
    # Safe bypasses — low-stakes, unproblematic claims
    (
        {"claim": "Proteins fold into specific three-dimensional structures.",
         "proposed_type": "concept", "importance": 0.3},
        True, True,
        "Safe concept bypass — factual, low-stakes",
    ),
    (
        {"claim": "Genes are inherited from parents.",
         "proposed_type": "concept", "importance": 0.4},
        True, True,
        "Safe concept bypass — basic fact",
    ),
    (
        {"claim": "Node X is associated with Node Y.",
         "edge_type": "associated", "importance": 0.5},
        True, True,
        "Safe associated edge bypass",
    ),
    (
        {"claim": "These concepts share surface similarities.",
         "edge_type": "surface_analogy", "importance": 0.6},
        True, True,
        "Safe surface analogy bypass",
    ),

    # Mandatory reviews — ALWAYS_REVIEW_TYPES should never be bypassed
    (
        {"claim": "A synthesis claim that must be reviewed.",
         "proposed_type": "synthesis", "importance": 0.3},
        False, False,
        "Synthesis must never bypass (ALWAYS_REVIEW_TYPES)",
    ),
    (
        {"claim": "A hypothesis that must be reviewed.",
         "proposed_type": "hypothesis", "importance": 0.2},
        False, False,
        "Hypothesis must never bypass (ALWAYS_REVIEW_TYPES)",
    ),
    (
        {"claim": "A structural analogy edge.",
         "edge_type": "structural_analogy", "importance": 0.3},
        False, False,
        "Structural analogy must never bypass (ALWAYS_REVIEW_TYPES)",
    ),
    (
        {"claim": "A deep isomorphism edge.",
         "edge_type": "deep_isomorphism", "importance": 0.2},
        False, False,
        "Deep isomorphism must never bypass (ALWAYS_REVIEW_TYPES)",
    ),

    # Edge cases: cross-domain and contradiction flags
    (
        {"claim": "Safe-looking concept, but crosses domains.",
         "proposed_type": "concept", "importance": 0.3,
         "crosses_domains": True},
        False, False,
        "Cross-domain flag should force review even for concepts",
    ),
    (
        {"claim": "Simple concept, but contradicts existing knowledge.",
         "proposed_type": "concept", "importance": 0.3,
         "contradicts_existing": True},
        False, False,
        "Contradiction flag should force review",
    ),

    # High importance concept (still bypassed — importance doesn't override BYPASS_TYPES)
    (
        {"claim": "Very important concept.",
         "proposed_type": "concept", "importance": 0.80},
        True, True,
        "High-importance concept — still safe to bypass (concepts are factual)",
    ),

    # Low importance, not in bypass or always-review types
    (
        {"claim": "A gap that is low importance.",
         "proposed_type": "gap", "importance": 0.3},
        True, True,
        "Low-importance gap — not in any special type list",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="benchmark/dim4/results/d4_bypass_safety.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 6: Bypass Safety (deterministic)")
    print("=" * 60)

    critic, _, _, _ = make_critic()

    evaluations = []
    safe_bypasses = 0
    total_bypasses = 0
    unsafe_bypass_count = 0
    mandatory_review_total = 0
    mandatory_review_compliant = 0

    for kwargs, should_bypass, is_safe, description in CASES:
        candidate = CandidateThought(**kwargs)
        actual_needs_review = critic.needs_review(candidate)
        actual_bypass = not actual_needs_review
        correct = actual_bypass == should_bypass

        # Track bypass safety
        if actual_bypass:
            total_bypasses += 1
            if is_safe:
                safe_bypasses += 1
            else:
                unsafe_bypass_count += 1

        # Track mandatory review compliance
        claim_type = kwargs.get("edge_type") or kwargs.get("proposed_type", "")
        from config import CRITIC as CRITIC_CFG
        if claim_type in CRITIC_CFG.ALWAYS_REVIEW_TYPES:
            mandatory_review_total += 1
            if actual_needs_review:
                mandatory_review_compliant += 1

        evaluations.append({
            "description": description,
            "claim": kwargs.get("claim", ""),
            "proposed_type": kwargs.get("proposed_type", ""),
            "edge_type": kwargs.get("edge_type", ""),
            "importance": kwargs.get("importance", 0.7),
            "should_bypass": should_bypass,
            "is_safe_to_bypass": is_safe,
            "actual_bypass": actual_bypass,
            "correct": correct,
        })

        status = "✓" if correct else "✗"
        label = "BYPASS" if actual_bypass else "REVIEW"
        print(f"  {status} {description}: {label}")

    safe_bypass_rate = safe_bypasses / max(total_bypasses, 1)
    mandatory_review_compliance = (
        mandatory_review_compliant / max(mandatory_review_total, 1)
    )

    passed = (
        safe_bypass_rate >= 0.90
        and mandatory_review_compliance == 1.0
    )

    report = {
        "test": "D4 - Bypass Safety",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"deterministic": True},
        "summary": {
            "cases_evaluated": len(CASES),
            "total_bypasses": total_bypasses,
            "safe_bypasses": safe_bypasses,
            "unsafe_bypass_count": unsafe_bypass_count,
            "safe_bypass_rate": round(safe_bypass_rate, 3),
            "mandatory_review_total": mandatory_review_total,
            "mandatory_review_compliant": mandatory_review_compliant,
            "mandatory_review_compliance": round(mandatory_review_compliance, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSafe bypass rate           : {safe_bypass_rate:.2%}")
    print(f"Unsafe bypasses            : {unsafe_bypass_count}")
    print(f"Mandatory review compliance: {mandatory_review_compliance:.2%}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
