"""
Dimension 6 - Test 2: Abstraction Quality
=========================================
Tests whether Consolidator._abstraction_pass() identifies a genuine higher-order
principle on real graph-derived node sets rather than assigning a cheap domain
label or a vacuous cross-domain slogan.
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

from _shared import get_isolated_graph, judge_json, override_models, resolve_suite_nodes
from consolidator.consolidator import ConsolidationReport, Consolidator

ABSTRACTION_JUDGE = """
You are evaluating whether a generated abstraction is a genuine META-PATTERN
or just a categorical label.

INPUT IDEAS:
{inputs}

GENERATED ABSTRACTION:
"{abstraction}"

Evaluate:
1. Meta-pattern: Does the abstraction describe a mechanism, organizing
   principle, or structural regularity explaining why the inputs belong
   together?
2. Specificity: Would this abstraction stop making sense if one of the inputs
   were swapped out for a random unrelated sentence? If it would still sound
   plausible for almost any arbitrary triad, it is too generic and should be
   treated as false.
3. Richness: Is it explanatory rather than a thin relabeling?
4. Input grounding: For each input separately, is there a concrete reason that
   THIS input belongs under the claimed meta-pattern?
5. Non-vacuity: Reject abstractions built mostly out of broad container words
   like "optimization", "constraint", "flow", "potential", or "adaptation"
   unless they are concretely grounded in all three inputs.

Score:
- quality: 1 to 7

Respond EXACTLY in JSON:
{{
  "genuine_pattern": true or false,
  "input_1_grounded": true or false,
  "input_2_grounded": true or false,
  "input_3_grounded": true or false,
  "non_vacuous": true or false,
  "quality": 1 to 7,
  "reasoning": "one sentence"
}}
"""

ABSTRACTION_SKEPTIC_JUDGE = """
You are the skeptical reviewer of a proposed abstraction.

INPUT IDEAS:
{inputs}

PROPOSED ABSTRACTION:
"{abstraction}"

Look for forced fits.

Evaluate:
1. Is any one input only weakly or metaphorically connected to the abstraction?
2. Could the abstraction still sound equally convincing if one of the inputs
   were replaced by a different sentence from an unrelated field?
3. Does the abstraction rely on broad words that hide a mismatch?

Respond EXACTLY in JSON:
{{
  "survives_adversarial_check": true or false,
  "forced_fit_present": true or false,
  "reasoning": "one sentence"
}}
"""

SUITES = [
    {
        "name": "genetic-physical-inheritance",
        "expect_pattern": True,
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
        "name": "thermodynamic-entropy-organization",
        "expect_pattern": True,
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
        "name": "neural-functional-organization",
        "expect_pattern": True,
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
    {
        "name": "thermo-game-neuro-mix",
        "expect_pattern": False,
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
        "name": "molecular-thermo-localization-mix",
        "expect_pattern": False,
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
        "name": "philosophy-molecular-game-mix",
        "expect_pattern": False,
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


def run_suite(generator_model: str, judge_model: str, suite: dict) -> dict:
    cluster_name = f"abstraction_case_{suite['name']}"

    with override_models(generator_model):
        brain, _ = get_isolated_graph()
        selected_ids = resolve_suite_nodes(brain, suite["selectors"])
        for nid in selected_ids:
            brain.update_node(nid, cluster=cluster_name, importance=0.95)

        consolidator = Consolidator(brain)
        before_ids = set(brain.graph.nodes)
        report = ConsolidationReport()
        consolidator._abstraction_pass(report)

    abstractions = []
    for nid, data in brain.all_nodes():
        if nid in before_ids:
            continue
        if data.get("cluster") == cluster_name:
            abstractions.append(data["statement"])

    inputs = [brain.get_node(nid)["statement"] for nid in selected_ids if brain.get_node(nid)]
    evaluations = []
    genuine_count = 0
    quality_values = []

    if abstractions:
        input_text = "\n".join([f"- {text}" for text in inputs])
        for abstraction in abstractions:
            judgment = judge_json(
                ABSTRACTION_JUDGE.format(inputs=input_text, abstraction=abstraction),
                model=judge_model,
                default={
                    "genuine_pattern": False,
                    "input_1_grounded": False,
                    "input_2_grounded": False,
                    "input_3_grounded": False,
                    "non_vacuous": False,
                    "quality": 1,
                    "reasoning": "Parse failed",
                },
            )
            skeptic = judge_json(
                ABSTRACTION_SKEPTIC_JUDGE.format(
                    inputs=input_text,
                    abstraction=abstraction,
                ),
                model=judge_model,
                default={
                    "survives_adversarial_check": False,
                    "forced_fit_present": True,
                    "reasoning": "Parse failed",
                },
            )
            evaluations.append(
                {
                    "statement": abstraction,
                    "judgment": judgment,
                    "skeptic_judgment": skeptic,
                }
            )
            if (
                judgment.get("genuine_pattern", False)
                and judgment.get("input_1_grounded", False)
                and judgment.get("input_2_grounded", False)
                and judgment.get("input_3_grounded", False)
                and judgment.get("non_vacuous", False)
                and skeptic.get("survives_adversarial_check", False)
                and not skeptic.get("forced_fit_present", True)
            ):
                genuine_count += 1
            quality_values.append(float(judgment.get("quality", 1)))

    if suite["expect_pattern"]:
        suite_pass = len(abstractions) > 0 and genuine_count > 0
    else:
        suite_pass = len(abstractions) == 0 or genuine_count == 0

    return {
        "suite": suite["name"],
        "expect_pattern": suite["expect_pattern"],
        "inputs": inputs,
        "abstractions_generated": len(abstractions),
        "genuine_count": genuine_count,
        "mean_quality": (
            sum(quality_values) / len(quality_values)
        ) if quality_values else 0.0,
        "PASS": suite_pass,
        "evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--out",
        default="benchmark/dim6/results/d6_abstraction_quality.json",
    )
    args = parser.parse_args()

    judge_model = args.judge_model or args.model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 2: Abstraction Quality")
    print("=" * 60)

    suite_results = []
    for suite in SUITES:
        print(f"\n  Suite: {suite['name']} (expect_pattern={suite['expect_pattern']})")
        result = run_suite(args.model, judge_model, suite)
        for statement in result["inputs"]:
            print(f"    - {statement}")
        suite_results.append(result)

    pos_results = [r for r in suite_results if r["expect_pattern"]]
    neg_results = [r for r in suite_results if not r["expect_pattern"]]
    pos_pass_rate = (
        sum(1 for r in pos_results if r["PASS"]) / len(pos_results)
        if pos_results else 0.0
    )
    neg_pass_rate = (
        sum(1 for r in neg_results if r["PASS"]) / len(neg_results)
        if neg_results else 0.0
    )
    quality_values = [
        float(r.get("mean_quality", 0.0))
        for r in suite_results
        if r.get("mean_quality", 0.0) > 0
    ]
    mean_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0

    passed = (
        pos_pass_rate >= 0.75
        and neg_pass_rate >= 1.0
        and mean_quality >= 4.0
    )
    summary = {
        "suite_count": len(suite_results),
        "positive_suite_count": len(pos_results),
        "negative_suite_count": len(neg_results),
        "positive_pass_rate": round(pos_pass_rate, 3),
        "negative_pass_rate": round(neg_pass_rate, 3),
        "quality": round(mean_quality, 3),
        "PASS": passed,
    }

    report_json = {
        "test": "D6 - Abstraction Quality",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"model": args.model, "judge_model": judge_model},
        "summary": summary,
        "suite_results": suite_results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    verdict = "PASS" if passed else "FAIL"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
