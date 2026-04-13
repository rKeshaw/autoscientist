"""
Dimension 5 - Test 2: Extraction Signal-to-Noise Ratio
=======================================================
Tests whether the Ingestor's node extraction produces meaningful
conceptual statements (signal) vs noise (too vague, keyword-like,
redundant, or malformed).

Uses the real `Ingestor.ingest()` path on curated text passages and an
LLM judge to evaluate each created node independently.

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

from _shared import make_ingestor, judge_json, get_fresh_brain


# ── Judge prompt ─────────────────────────────────────────────────────────────

NODE_QUALITY_JUDGE = """
You are evaluating the quality of a node extracted for a scientific knowledge graph.

SOURCE TEXT (excerpt):
"{source_text}"

EXTRACTED NODE:
"{node}"

Evaluate this extracted node on three dimensions:

1. **Self-contained**: Is the node a complete thought that stands alone?
   (vs a keyword, topic label, or fragment)
2. **Informativeness**: Does it capture a specific insight, mechanism, or relationship?
   (vs being vague, generic, or trivially obvious)
3. **Fidelity**: Is the content faithful to the source text?
   (vs hallucinated, distorted, or over-interpreted)

Score each 1-7 and classify as "signal" or "noise":
- signal: A genuine, useful conceptual statement worth adding to a knowledge graph
- noise: Too vague, keyword-like, redundant, trivial, or malformed to be useful

Respond EXACTLY in JSON:
{{
  "classification": "signal" or "noise",
  "self_contained": 1 to 7,
  "informativeness": 1 to 7,
  "fidelity": 1 to 7,
  "reasoning": "one sentence"
}}
"""


# ── Test passages ────────────────────────────────────────────────────────────

PASSAGES = [
    {
        "title": "Synaptic Plasticity",
        "text": (
            "Synaptic plasticity is the ability of synapses to strengthen or weaken "
            "over time, in response to increases or decreases in their activity. "
            "Long-term potentiation (LTP) and long-term depression (LTD) are widely "
            "considered to be the major cellular mechanisms underlying learning and "
            "memory. LTP is a long-lasting enhancement in signal transmission between "
            "two neurons, following synchronous stimulation. It is one of several "
            "phenomena underlying synaptic plasticity. Hebbian theory proposes that "
            "synaptic connections are strengthened when a presynaptic neuron repeatedly "
            "and persistently stimulates a postsynaptic neuron, often summarized as "
            "'cells that fire together, wire together.'"
        ),
    },
    {
        "title": "Genetic Algorithms",
        "text": (
            "A genetic algorithm (GA) is a metaheuristic inspired by the process of "
            "natural selection. Genetic algorithms are commonly used to generate high-"
            "quality solutions to optimization and search problems by relying on "
            "biologically inspired operators such as mutation, crossover and selection. "
            "The fitness function evaluates how close a given solution is to the optimum "
            "and guides the selection of candidates for reproduction. Elitism ensures "
            "that the best solutions carry over to the next generation, preventing "
            "regression in overall fitness."
        ),
    },
    {
        "title": "Information Entropy",
        "text": (
            "In information theory, the entropy of a random variable is the average "
            "level of 'information', 'surprise', or 'uncertainty' inherent to the "
            "variable's possible outcomes. Given a discrete random variable X with "
            "possible outcomes x1, ..., xn, the entropy H(X) is defined as the "
            "negative sum of p(xi) log p(xi). The concept was introduced by Claude "
            "Shannon in his 1948 paper 'A Mathematical Theory of Communication'. "
            "Entropy is maximized when all outcomes are equally likely, and is zero "
            "when one outcome is certain."
        ),
    },
    {
        "title": "CRISPR-Cas9",
        "text": (
            "CRISPR-Cas9 is a genome editing tool that allows researchers to alter "
            "DNA sequences and modify gene function. The technology was adapted from "
            "the natural immune defense mechanisms of bacteria, which use CRISPR-"
            "derived RNA and Cas9 protein to detect and destroy DNA of invading "
            "viruses. The guide RNA (gRNA) directs the Cas9 enzyme to the target "
            "site in the genome, where it creates a double-strand break. The cell's "
            "repair machinery then introduces insertions or deletions, effectively "
            "knocking out the gene."
        ),
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim5/results/d5_extraction_snr.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 2: Extraction Signal-to-Noise Ratio")
    print("=" * 60)

    from graph.brain import EdgeSource

    evaluations = []
    total_nodes = 0
    signal_count = 0
    total_quality = 0.0
    passages_with_zero_nodes = 0

    for passage in PASSAGES:
        print(f"\n  Passage: {passage['title']}")

        brain, emb_index = get_fresh_brain()
        ingestor, brain, emb_index, _ = make_ingestor(
            brain=brain, emb_index=emb_index, mission=None
        )
        node_ids = ingestor.ingest(passage["text"], source=EdgeSource.READING) or []

        clean_nodes = []
        for node_id in node_ids:
            node = brain.get_node(node_id)
            if node and node.get("statement"):
                clean_nodes.append(node["statement"].strip())

        print(f"    Extracted {len(clean_nodes)} nodes")
        if not clean_nodes:
            passages_with_zero_nodes += 1

        passage_evals = []
        for node_text in clean_nodes:
            total_nodes += 1

            judgment = judge_json(
                NODE_QUALITY_JUDGE.format(
                    source_text=passage["text"][:500],
                    node=node_text,
                ),
                model=args.judge_model,
                default={
                    "classification": "noise",
                    "self_contained": 1,
                    "informativeness": 1,
                    "fidelity": 1,
                    "reasoning": "Judge parse failed",
                },
            )

            is_signal = judgment.get("classification", "noise") == "signal"
            quality = (
                float(judgment.get("self_contained", 1))
                + float(judgment.get("informativeness", 1))
                + float(judgment.get("fidelity", 1))
            ) / 3.0

            if is_signal:
                signal_count += 1
            total_quality += quality

            passage_evals.append({
                "node": node_text[:200],
                "classification": "signal" if is_signal else "noise",
                "quality": round(quality, 3),
                "judgment": judgment,
            })

            status = "✓" if is_signal else "✗"
            print(f"    {status} [{quality:.1f}/7] {node_text[:80]}...")

        evaluations.append({
            "title": passage["title"],
            "text_length": len(passage["text"]),
            "nodes_extracted": len(clean_nodes),
            "signal_count": sum(1 for e in passage_evals if e["classification"] == "signal"),
            "nodes": passage_evals,
        })
        time.sleep(0.2)

    n = max(total_nodes, 1)
    signal_fraction = signal_count / n
    mean_quality = total_quality / n

    passed = (
        total_nodes > 0 and
        passages_with_zero_nodes == 0 and
        signal_fraction >= 0.75 and
        mean_quality >= 4.5
    )

    report = {
        "test": "D5 - Extraction SNR",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model},
        "summary": {
            "passages_evaluated": len(PASSAGES),
            "total_nodes_extracted": total_nodes,
            "signal_count": signal_count,
            "signal_fraction": round(signal_fraction, 3),
            "mean_quality": round(mean_quality, 3),
            "passages_with_zero_nodes": passages_with_zero_nodes,
            "benchmark_exercised": total_nodes > 0,
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSignal fraction: {signal_fraction:.2%}")
    print(f"Mean quality   : {mean_quality:.2f}/7")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
