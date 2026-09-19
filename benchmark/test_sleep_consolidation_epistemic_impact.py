#!/usr/bin/env python3
"""
test_sleep_consolidation_epistemic_impact.py

Phase 2 Experiment: Causal Impact of Sleep Consolidation on Scientific Reasoning.
Compares the full AutoScientist brain graph against a counterfactual
"sleep-deprived" graph (where synthesis and structural gap consolidation passes are ablated).

Evaluates across the 24-Task Epistemic Benchmark:
1. Intact Brain Graph (With Sleep Consolidation - Full Passes 1-6)
2. Sleep-Deprived Brain Graph (No Synthesis, No Gaps - Ablation)

Metrics:
- Pass Rate per category (Multi-Hop Analogy, Contradictions, Empirical, Negative Knowledge)
- Multi-Hop Bridge Discovery Rate
- Mean Keyword Recall and Mean Reciprocal Rank (MRR)
- Cross-Domain Shortest Path Distance between distant scientific concepts
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import re
import math
import numpy as np
import networkx as nx

from embedding_index import EmbeddingIndex
from embedding import embed as shared_embed
from benchmark.test_representation_baselines_expanded import (
    BrainGraphSubstrate, EXPANDED_BENCHMARK_TASKS, evaluate_task_retrieval
)

BRAIN_PATH = "data/brain.json"
INDEX_PATH = "data/embedding_index"
OUTPUT_DIR = "benchmark/results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "sleep_consolidation_ablation_results.json")


def build_ablated_brain_data(master_brain: dict, exclude_types: set) -> dict:
    """Create a counterfactual brain graph copy omitting specific node types."""
    ablated = {"graph": {"nodes": [], "edges": []}}
    valid_ids = set()

    for n in master_brain["graph"]["nodes"]:
        if n.get("node_type") not in exclude_types:
            valid_ids.add(n["id"])
            ablated["graph"]["nodes"].append(n)

    for e in master_brain["graph"]["edges"]:
        if e["source"] in valid_ids and e["target"] in valid_ids:
            ablated["graph"]["edges"].append(e)

    return ablated


def run_sleep_ablation_benchmark():
    print("=" * 80)
    print("AUTOSCIENTIST: SLEEP CONSOLIDATION ABLATION BENCHMARK (PHASE 2)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(BRAIN_PATH, 'r') as f:
        master_brain = json.load(f)

    emb_index = EmbeddingIndex.load(INDEX_PATH)

    print("Building substrates:")
    print("  1. Condition A: Full Brain Graph (With Sleep Consolidation)")
    full_sub = BrainGraphSubstrate(master_brain, emb_index)

    print("  2. Condition B: Sleep-Deprived Brain Graph (No Synthesis, No Gaps)")
    ablated_data = build_ablated_brain_data(master_brain, exclude_types={"synthesis", "gap"})
    sleep_deprived_sub = BrainGraphSubstrate(ablated_data, emb_index)

    print(f"  Full Graph: {full_sub.G.number_of_nodes()} nodes, {full_sub.G.number_of_edges()} edges")
    print(f"  Sleep-Deprived: {sleep_deprived_sub.G.number_of_nodes()} nodes, {sleep_deprived_sub.G.number_of_edges()} edges\n")

    conditions = {
        "Full (With Sleep)": full_sub,
        "Sleep-Deprived (Ablated)": sleep_deprived_sub
    }

    report = {"tasks": [], "by_category": {}, "aggregate": {}}
    categories = ["multi_hop_analogy", "contradiction_preservation", "empirical_grounding", "negative_knowledge"]

    for idx, task in enumerate(EXPANDED_BENCHMARK_TASKS, start=1):
        cat = task["category"]
        tid = task["task_id"]
        task_record = {"task_id": tid, "category": cat, "conditions": {}}

        for name, sub in conditions.items():
            results = sub.retrieve(task["query"], top_k=6)
            eval_res = evaluate_task_retrieval(results, task)
            task_record["conditions"][name] = eval_res

        status_full = "PASS" if task_record["conditions"]["Full (With Sleep)"]["success"] else "FAIL"
        status_abl = "PASS" if task_record["conditions"]["Sleep-Deprived (Ablated)"]["success"] else "FAIL"
        delta_rec = task_record["conditions"]["Full (With Sleep)"]["keyword_recall"] - task_record["conditions"]["Sleep-Deprived (Ablated)"]["keyword_recall"]

        print(f"Task {idx:02d}/24 [{cat}]: {tid}")
        print(f"  Full: {status_full} (Rec: {task_record['conditions']['Full (With Sleep)']['keyword_recall']:.2f}) | "
              f"Sleep-Deprived: {status_abl} (Rec: {task_record['conditions']['Sleep-Deprived (Ablated)']['keyword_recall']:.2f}) | "
              f"Delta: {delta_rec:+.2f}")

        report["tasks"].append(task_record)

    # Category summaries
    for cat in categories:
        cat_tasks = [t for t in report["tasks"] if t["category"] == cat]
        report["by_category"][cat] = {}
        for name in conditions:
            n_cat = len(cat_tasks)
            passed = sum(1 for t in cat_tasks if t["conditions"][name]["success"])
            mean_rec = np.mean([t["conditions"][name]["keyword_recall"] for t in cat_tasks])
            mean_mrr = np.mean([t["conditions"][name]["mrr"] for t in cat_tasks])
            report["by_category"][cat][name] = {
                "pass_rate": round(passed / n_cat, 3),
                "passed": passed,
                "total": n_cat,
                "mean_recall": round(float(mean_rec), 3),
                "mean_mrr": round(float(mean_mrr), 3)
            }

    # Aggregate summaries
    for name in conditions:
        total = len(EXPANDED_BENCHMARK_TASKS)
        passed = sum(1 for t in report["tasks"] if t["conditions"][name]["success"])
        mean_rec = np.mean([t["conditions"][name]["keyword_recall"] for t in report["tasks"]])
        mean_mrr = np.mean([t["conditions"][name]["mrr"] for t in report["tasks"]])
        report["aggregate"][name] = {
            "pass_rate": round(passed / total, 3),
            "passed": passed,
            "total": total,
            "mean_recall": round(float(mean_rec), 3),
            "mean_mrr": round(float(mean_mrr), 3)
        }

    print("\n" + "=" * 80)
    print("SLEEP CONSOLIDATION ABLATION SUMMARY:")
    print("=" * 80)
    for name, stats in report["aggregate"].items():
        print(f"  {name:30s} | Pass Rate: {stats['pass_rate']*100:5.1f}% ({stats['passed']}/{stats['total']}) | "
              f"Recall: {stats['mean_recall']:.3f} | MRR: {stats['mean_mrr']:.3f}")

    print("\nCATEGORY DELTAS (Impact of Sleep Deprivation):")
    for cat in categories:
        s_full = report["by_category"][cat]["Full (With Sleep)"]
        s_abl = report["by_category"][cat]["Sleep-Deprived (Ablated)"]
        pass_delta = (s_full["pass_rate"] - s_abl["pass_rate"]) * 100
        rec_delta = s_full["mean_recall"] - s_abl["mean_recall"]
        print(f"  [{cat:28s}] Pass Delta: {pass_delta:+5.1f}% | Recall Delta: {rec_delta:+.3f}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved detailed Phase 2 ablation report to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_sleep_ablation_benchmark()
