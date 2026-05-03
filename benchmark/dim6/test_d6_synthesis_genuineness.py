"""
Dimension 6 - Test 1: Synthesis Genuineness
===========================================
Tests whether Consolidator._synthesis_pass() produces genuinely emergent
cross-node ideas on the inherited benchmark graph, abstains on incoherent real
node sets, and avoids counting field-overview restatements as valid syntheses.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import (
    get_isolated_graph,
    judge_json,
    override_models,
    pairwise_similarity,
    resolve_suite_nodes,
)
from config import THRESHOLDS
from consolidator.consolidator import ConsolidationReport, Consolidator

SYNTHESIS_JUDGE = """
You are evaluating whether a generated synthesis is a genuinely NEW idea that
emerges from a set of inputs, or just a summary.

INPUT IDEAS:
{inputs}

GENERATED SYNTHESIS:
"{synthesis}"

Evaluate:
1. Emergence: Does the synthesis offer a conclusion, mechanism, or explanatory
   insight that is not explicitly stated in any single input?
2. Coherence: Is the synthesis logically grounded in the inputs rather than a
   free hallucination?

Score:
- quality: 1 to 7

Respond EXACTLY in JSON:
{{
  "genuine": true or false,
  "quality": 1 to 7,
  "reasoning": "one sentence"
}}
"""

POSITIVE_SUITES = [
    {
        "name": "genetic-physical-substrate",
        "selectors": [
            {
                "cluster": "genetics",
                "contains_all": [
                    "dna is the molecule responsible",
                    "transmitting genetic information",
                ],
            },
            {
                "cluster": "genetics",
                "contains_all": [
                    "genes responsible for heredity",
                    "located on chromosomes",
                ],
            },
            {
                "cluster": "genetics",
                "contains_all": [
                    "sex-linked mutations",
                    "linearly arranged loci on chromosomes",
                ],
            },
        ],
    },
    {
        "name": "thermodynamic-entropy-bridge",
        "selectors": [
            {
                "cluster": "thermodynamics",
                "contains_all": ["first law", "second law"],
            },
            {
                "cluster": "thermodynamics",
                "contains_all": [
                    "average motion of its constituent particles",
                    "equations of state",
                ],
            },
            {
                "cluster": "thermodynamics",
                "contains_all": [
                    "information-theoretic entropy",
                    "classical thermodynamic entropy",
                ],
            },
        ],
    },
    {
        "name": "evolutionary-innovation",
        "selectors": [
            {
                "cluster": "evolutionary_biology",
                "contains_all": [
                    "natural selection is defined",
                    "relative fitness",
                ],
            },
            {
                "cluster": "evolutionary_biology",
                "contains_all": ["modern synthesis", "darwinian principles"],
            },
            {
                "cluster": "evolutionary_biology",
                "contains_all": ["gene duplication", "novel functions"],
            },
        ],
    },
    {
        "name": "neural-functional-architecture",
        "selectors": [
            {
                "cluster": "neuroscience",
                "contains_all": [
                    "neuron is the discrete",
                    "functional unit of the nervous system",
                ],
            },
            {
                "cluster": "neuroscience",
                "contains_all": [
                    "action potentials in axons",
                    "mathematical modeling",
                ],
            },
            {
                "cluster": "neuroscience",
                "contains_all": [
                    "localized lesion studies",
                    "specific cognitive and motor skills",
                ],
            },
        ],
    },
]

INCOHERENT_NEGATIVE_SUITES = [
    {
        "name": "thermo-game-neuro",
        "selectors": [
            {
                "cluster": "thermodynamics",
                "contains_all": ["first law", "second law"],
            },
            {
                "cluster": "game_theory",
                "contains_all": ["nash equilibrium", "mixed-strategy equilibrium"],
            },
            {
                "cluster": "neuroscience",
                "contains_all": [
                    "neuron is the discrete",
                    "functional unit of the nervous system",
                ],
            },
        ],
    },
    {
        "name": "molecular-thermo-localization",
        "selectors": [
            {
                "cluster": "molecular_biology",
                "contains_all": [
                    "molecular substrate responsible",
                    "transferring genetic material",
                ],
            },
            {
                "cluster": "thermodynamics",
                "contains_all": [
                    "interrelations among heat, work, and temperature",
                    "quantitative description of energy and entropy",
                ],
            },
            {
                "cluster": "neuroscience",
                "contains_all": [
                    "localized to distinct",
                    "cerebral cortex",
                ],
            },
        ],
    },
    {
        "name": "info-evo-neurohistory",
        "selectors": [
            {
                "cluster": "information_theory",
                "contains_all": [
                    "degree of uncertainty",
                    "information gain",
                ],
            },
            {
                "cluster": "evolutionary_biology",
                "contains_all": ["gene duplication", "novel functions"],
            },
            {
                "cluster": "neuroscience",
                "contains_all": [
                    "heart constituted the seat of intelligence",
                    "recognition",
                ],
            },
        ],
    },
    {
        "name": "philosophy-molecular-game",
        "selectors": [
            {
                "cluster": "philosophy_of_science",
                "contains_all": ["natural teleology", "for a purpose"],
            },
            {
                "cluster": "molecular_biology",
                "contains_all": [
                    "noncanonical bases in bacterial viruses",
                    "restriction enzymes",
                ],
            },
            {
                "cluster": "game_theory",
                "contains_all": ["cournot", "price competition"],
            },
        ],
    },
]

BOUNDARY_SUITES = [
    {
        "name": "genetics-field-overview",
        "selectors": [
            {
                "cluster": "genetics",
                "contains_all": [
                    "genetics fundamentally investigates",
                    "mechanisms of heredity",
                ],
            },
            {
                "cluster": "genetics",
                "contains_all": [
                    "modern genetics encompasses",
                    "multiple scales",
                ],
            },
            {
                "cluster": "genetics",
                "contains_all": [
                    "study of inheritance has transitioned",
                    "rigorous scientific discipline",
                ],
            },
        ],
    },
]


def _suite_inputs(brain, node_ids: list[str]) -> list[str]:
    return [brain.get_node(nid)["statement"] for nid in node_ids if brain.get_node(nid)]


def evaluate_suite(
    generator_model: str,
    judge_model: str,
    selectors: list[dict],
    expected_behavior: str,
):
    with override_models(generator_model):
        brain, _ = get_isolated_graph()
        consolidator = Consolidator(brain)
        node_ids = resolve_suite_nodes(brain, selectors)
        inputs = _suite_inputs(brain, node_ids)
        avg_sim = pairwise_similarity(consolidator, node_ids)

        report = ConsolidationReport()
        consolidator._synthesis_pass(node_ids, report)

    syntheses = []
    for nid in report.synthesis_ids:
        data = brain.get_node(nid)
        if data:
            syntheses.append(data["statement"])

    if expected_behavior == "genuine" and not syntheses:
        failure_mode = (
            "low_cohesion_gate"
            if avg_sim < THRESHOLDS.SYNTHESIS_COHESION
            else "llm_abstained_or_parse_failed"
        )
        return {
            "inputs": inputs,
            "avg_pair_similarity": round(avg_sim, 3),
            "synthesized": False,
            "genuine": False,
            "quality": 0.0,
            "failure_mode": failure_mode,
            "reasoning": "No synthesis node generated for a positive suite.",
            "evaluations": [],
            "PASS": False,
        }

    if expected_behavior in {"abstain", "nongenuine"} and not syntheses:
        return {
            "inputs": inputs,
            "avg_pair_similarity": round(avg_sim, 3),
            "synthesized": False,
            "genuine": False,
            "quality": 7.0,
            "failure_mode": "abstained",
            "reasoning": (
                "Correctly abstained on an incoherent input set."
                if expected_behavior == "abstain"
                else "Safely abstained on a summary-like boundary case."
            ),
            "evaluations": [],
            "PASS": True,
        }

    evaluations = []
    genuine_count = 0
    quality_values = []
    input_text = "\n".join([f"- {text}" for text in inputs])

    for synthesis in syntheses:
        judgment = judge_json(
            SYNTHESIS_JUDGE.format(inputs=input_text, synthesis=synthesis),
            model=judge_model,
            default={"genuine": False, "quality": 1, "reasoning": "Parse failed"},
        )
        evaluations.append({"statement": synthesis, "judgment": judgment})
        if judgment.get("genuine", False):
            genuine_count += 1
        quality_values.append(float(judgment.get("quality", 1)))

    mean_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0
    if expected_behavior == "genuine":
        suite_pass = genuine_count > 0
        reasoning = "At least one generated synthesis was judged emergent."
    elif expected_behavior == "abstain":
        suite_pass = False
        reasoning = "Generated a synthesis where the benchmark expected abstention."
    else:
        suite_pass = genuine_count == 0
        reasoning = (
            "Generated text was safely judged non-emergent for the summary-like boundary case."
            if suite_pass
            else "A summary-like boundary case produced a synthesis judged genuinely novel."
        )
    return {
        "inputs": inputs,
        "avg_pair_similarity": round(avg_sim, 3),
        "synthesized": True,
        "genuine": genuine_count > 0,
        "quality": mean_quality,
        "failure_mode": None,
        "reasoning": reasoning,
        "evaluations": evaluations,
        "PASS": suite_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--out",
        default="benchmark/dim6/results/d6_synthesis_genuineness.json",
    )
    args = parser.parse_args()

    judge_model = args.judge_model or args.model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 1: Synthesis Genuineness")
    print("=" * 60)

    positive_results = []
    incoherent_negative_results = []
    boundary_results = []

    for suite in POSITIVE_SUITES:
        print(f"\n  Positive suite: {suite['name']}")
        result = evaluate_suite(args.model, judge_model, suite["selectors"], "genuine")
        for statement in result["inputs"]:
            print(f"    - {statement}")
        positive_results.append({"suite": suite["name"], "result": result})

    for suite in INCOHERENT_NEGATIVE_SUITES:
        print(f"\n  Incoherent negative suite: {suite['name']}")
        result = evaluate_suite(args.model, judge_model, suite["selectors"], "abstain")
        for statement in result["inputs"]:
            print(f"    - {statement}")
        incoherent_negative_results.append({"suite": suite["name"], "result": result})

    for suite in BOUNDARY_SUITES:
        print(f"\n  Boundary suite: {suite['name']}")
        result = evaluate_suite(args.model, judge_model, suite["selectors"], "nongenuine")
        for statement in result["inputs"]:
            print(f"    - {statement}")
        boundary_results.append({"suite": suite["name"], "result": result})

    pos_total = len(positive_results)
    neg_total = len(incoherent_negative_results)
    boundary_total = len(boundary_results)
    pos_genuine = sum(1 for r in positive_results if r["result"].get("PASS", False))
    neg_abstained = sum(
        1 for r in incoherent_negative_results if r["result"].get("PASS", False)
    )
    boundary_safe = sum(1 for r in boundary_results if r["result"].get("PASS", False))
    pos_quality_values = [
        float(r["result"].get("quality", 0.0))
        for r in positive_results
        if r["result"].get("synthesized", False)
    ]

    pos_genuine_rate = pos_genuine / pos_total if pos_total else 0.0
    neg_abstain_rate = neg_abstained / neg_total if neg_total else 0.0
    boundary_safe_rate = boundary_safe / boundary_total if boundary_total else 1.0
    avg_positive_quality = (
        sum(pos_quality_values) / len(pos_quality_values)
        if pos_quality_values else 0.0
    )

    passed = (
        pos_genuine_rate >= 0.75
        and neg_abstain_rate >= 1.0
        and boundary_safe_rate >= 1.0
        and avg_positive_quality >= 4.5
    )

    summary = {
        "positive_total": pos_total,
        "positive_genuine": pos_genuine,
        "positive_genuine_rate": round(pos_genuine_rate, 3),
        "positive_quality": round(avg_positive_quality, 3),
        "incoherent_negative_total": neg_total,
        "incoherent_negative_abstained": neg_abstained,
        "incoherent_negative_abstain_rate": round(neg_abstain_rate, 3),
        "boundary_total": boundary_total,
        "boundary_safe": boundary_safe,
        "boundary_safe_rate": round(boundary_safe_rate, 3),
        "PASS": passed,
    }

    report = {
        "test": "D6 - Synthesis Genuineness",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"model": args.model, "judge_model": judge_model},
        "summary": summary,
        "positive_suites": positive_results,
        "incoherent_negative_suites": incoherent_negative_results,
        "boundary_suites": boundary_results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    verdict = "PASS" if passed else "FAIL"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
