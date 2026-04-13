"""
Dimension 6 - Test 3: Gap Inference Accuracy
============================================
Tests whether Consolidator._gap_detection() infers a genuine mediating gap
from graph-derived node pairs taken out of the inherited benchmark brain.
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

from _shared import get_isolated_graph, judge_json, override_models, resolve_node_id
from consolidator.consolidator import ConsolidationReport, Consolidator
from graph.brain import Edge, EdgeSource, EdgeType

GAP_JUDGE = """
You are evaluating whether a generated gap is a genuinely necessary missing
link between two connected ideas.

IDEA A: "{node_a}"
IDEA B: "{node_b}"
NARRATION OF THEIR CONNECTION: "{narration}"

INFERRED GAP:
"{gap}"

Evaluate:
1. Missing link: Does the gap explain how or why the two ideas connect?
2. Specificity: Is it a real mediating mechanism rather than a broad related
   topic that happens to live nearby?
3. Non-redundancy: Is the gap more than a paraphrase, definition, or slightly
   more detailed restatement of either endpoint?

Score:
- quality: 1 to 7

Respond EXACTLY in JSON:
{{
  "genuine_link": true or false,
  "non_redundant": true or false,
  "quality": 1 to 7,
  "reasoning": "one sentence"
}}
"""

SUITES = [
    {
        "name": "environment-to-epigenetics",
        "expect_gap": True,
        "selector_a": {
            "cluster": "genetics",
            "contains_all": ["environmental conditions", "phenotypic expression"],
        },
        "selector_b": {
            "cluster": "genetics",
            "contains_all": ["epigenetics denotes changes in gene expression", "dna nucleotide sequence"],
        },
        "narration": "Environmental conditions can alter phenotype through mechanisms that do not rewrite the DNA sequence.",
        "edge_type": EdgeType.SUPPORTS,
    },
    {
        "name": "dna-to-chromosomal-genes",
        "expect_gap": True,
        "selector_a": {
            "cluster": "genetics",
            "contains_all": ["dna is the molecule responsible", "transmitting genetic information"],
        },
        "selector_b": {
            "cluster": "genetics",
            "contains_all": ["genes responsible for heredity", "located on chromosomes"],
        },
        "narration": "Modern genetics ties hereditary information to organized chromosomal structure.",
        "edge_type": EdgeType.SUPPORTS,
    },
    {
        "name": "selection-to-duplication",
        "expect_gap": True,
        "selector_a": {
            "cluster": "evolutionary_biology",
            "contains_all": ["natural selection is defined", "relative fitness"],
        },
        "selector_b": {
            "cluster": "evolutionary_biology",
            "contains_all": ["gene duplication", "novel functions"],
        },
        "narration": "Evolutionary innovation requires both selection pressure and mechanisms that generate new heritable variants.",
        "edge_type": EdgeType.SUPPORTS,
    },
    {
        "name": "entropy-to-channel-capacity",
        "expect_gap": True,
        "selector_a": {
            "cluster": "information_theory",
            "contains_all": ["degree of uncertainty", "information gain"],
        },
        "selector_b": {
            "cluster": "information_theory",
            "contains_all": ["noisy-channel coding theorem", "channel capacity"],
        },
        "narration": "Reliable communication under noise depends on quantifying uncertainty and then coding against it.",
        "edge_type": EdgeType.SUPPORTS,
    },
    {
        "name": "thermo-game-unrelated",
        "expect_gap": False,
        "selector_a": {
            "cluster": "thermodynamics",
            "contains_all": ["first law", "second law"],
        },
        "selector_b": {
            "cluster": "game_theory",
            "contains_all": ["nash equilibrium", "mixed-strategy equilibrium"],
        },
        "narration": "Both are formal theoretical frameworks used in science.",
        "edge_type": EdgeType.ASSOCIATED,
    },
    {
        "name": "neuron-doctrine-paraphrase",
        "expect_gap": False,
        "selector_a": {
            "cluster": "neuroscience",
            "contains_all": ["neuron is the discrete", "functional unit of the nervous system"],
        },
        "selector_b": {
            "cluster": "neuroscience",
            "contains_all": ["functional unit of the brain", "neuron"],
        },
        "narration": "Both statements make the same core point about the neuron as the basic unit.",
        "edge_type": EdgeType.SUPPORTS,
    },
    {
        "name": "genetics-discipline-overview",
        "expect_gap": False,
        "selector_a": {
            "cluster": "genetics",
            "contains_all": ["genetics fundamentally investigates", "mechanisms of heredity"],
        },
        "selector_b": {
            "cluster": "genetics",
            "contains_all": ["study of inheritance has transitioned", "rigorous scientific discipline"],
        },
        "narration": "Both statements are broad field overviews of how genetics is framed historically and scientifically.",
        "edge_type": EdgeType.SUPPORTS,
    },
]


def run_suite(generator_model: str, judge_model: str, suite: dict) -> dict:
    with override_models(generator_model):
        brain, _ = get_isolated_graph()
        n1_id = resolve_node_id(brain, **suite["selector_a"])
        n2_id = resolve_node_id(brain, **suite["selector_b"])

        for nid in list(brain.graph.nodes):
            if nid not in {n1_id, n2_id}:
                brain.graph.remove_node(nid)

        # Assign different temporary clusters so gap detection does not skip the
        # pair under the cross-cluster preference branch.
        brain.update_node(n1_id, cluster=f"{suite['name']}_a")
        brain.update_node(n2_id, cluster=f"{suite['name']}_b")

        brain.add_edge(
            n1_id,
            n2_id,
            Edge(
                type=suite["edge_type"],
                narration=suite["narration"],
                weight=1.0,
                confidence=0.9,
                source=EdgeSource.CONVERSATION,
            ),
        )

        consolidator = Consolidator(brain)
        report = ConsolidationReport()
        consolidator._gap_detection(report)

    node_a = brain.get_node(n1_id)["statement"]
    node_b = brain.get_node(n2_id)["statement"]
    gaps = []
    for nid in report.gap_ids:
        data = brain.get_node(nid)
        if data:
            gaps.append(data["statement"])

    evaluations = []
    genuine_count = 0
    quality_values = []
    for gap in gaps:
        judgment = judge_json(
            GAP_JUDGE.format(
                node_a=node_a,
                node_b=node_b,
                narration=suite["narration"],
                gap=gap,
            ),
            model=judge_model,
            default={
                "genuine_link": False,
                "non_redundant": False,
                "quality": 1,
                "reasoning": "Parse failed",
            },
        )
        evaluations.append({"statement": gap, "judgment": judgment})
        if judgment.get("genuine_link", False) and judgment.get("non_redundant", False):
            genuine_count += 1
        quality_values.append(float(judgment.get("quality", 1)))

    if suite["expect_gap"]:
        suite_pass = len(gaps) > 0 and genuine_count > 0
    else:
        suite_pass = len(gaps) == 0 or genuine_count == 0

    return {
        "suite": suite["name"],
        "expect_gap": suite["expect_gap"],
        "node_a": node_a,
        "node_b": node_b,
        "gaps_inferred": len(gaps),
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
        default="benchmark/dim6/results/d6_gap_inference.json",
    )
    args = parser.parse_args()

    judge_model = args.judge_model or args.model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 3: Gap Inference Accuracy")
    print("=" * 60)

    suite_results = []
    for suite in SUITES:
        print(f"\n  Suite: {suite['name']} (expect_gap={suite['expect_gap']})")
        result = run_suite(args.model, judge_model, suite)
        print(f"    - A: {result['node_a']}")
        print(f"    - B: {result['node_b']}")
        print(f"    - Edge: {suite['edge_type'].value} | {suite['narration']}")
        suite_results.append(result)

    pos_results = [r for r in suite_results if r["expect_gap"]]
    neg_results = [r for r in suite_results if not r["expect_gap"]]
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
        "test": "D6 - Gap Inference Accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "model": args.model,
            "judge_model": judge_model,
            "graph_source": "benchmark/dim4/shared/brain.json",
        },
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
