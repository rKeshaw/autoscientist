#!/usr/bin/env python3
"""
test_representation_baselines.py

Phase 1 Harness: Head-to-head empirical comparison of 3 memory substrates:
1. Brain-as-Graph (Condition C): Multi-relational graph with typed edge traversal
2. Flat Vector Store (Baseline B / RAG): Flat FAISS top-k similarity retrieval
3. Flat-File Memory (Baseline A / AutoScientists): Append-only linear markdown file with BM25 / keyword chunking

Evaluates:
- Multi-hop cross-domain analogy discovery
- Contradiction & scientific tension preservation
- Empirical evidence & negative result grounding
"""

import os
import sys

# Ensure repository root is on PYTHONPATH
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

BRAIN_PATH = "data/brain.json"
INDEX_PATH = "data/embedding_index"
OUTPUT_DIR = "benchmark/results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "representation_comparison.json")


# ── Substrate Wrappers ────────────────────────────────────────────────────────

class BrainGraphSubstrate:
    """Condition C: Brain-as-Graph with multi-hop typed relational traversal."""
    def __init__(self, brain_data: dict, emb_index: EmbeddingIndex):
        self.brain_data = brain_data
        self.emb_index = emb_index
        self.G = nx.DiGraph()
        self.nodes_dict = {}

        for n in brain_data["graph"]["nodes"]:
            nid = n["id"]
            self.nodes_dict[nid] = n
            self.G.add_node(nid, **{k: v for k, v in n.items() if k != "id"})

        for e in brain_data["graph"]["edges"]:
            src = e["source"]
            tgt = e["target"]
            self.G.add_edge(src, tgt, **{k: v for k, v in e.items() if k not in ("source", "target")})

    def retrieve(self, query: str, top_k: int = 5, hop_depth: int = 2) -> list:
        # Neurobiological Spreading Activation retrieval seeded by query vector
        q_vec = shared_embed(query)
        seed_matches = self.emb_index.query(q_vec, threshold=0.3, top_k=5)

        priority_edge_types = {
            "structural_analogy", "deep_isomorphism", "contradicts",
            "empirically_tested", "supports", "causes", "synthesis"
        }

        # Initialize activation energy at seed nodes
        activation = {}
        for nid, sim in seed_matches:
            if nid in self.G:
                activation[nid] = float(sim)

        # Spread activation across graph synapses
        for hop in range(hop_depth):
            current_nodes = list(activation.items())
            for src, act in current_nodes:
                if act < 0.1:
                    continue
                neighbors = list(self.G.successors(src)) + list(self.G.predecessors(src))
                for nbr in neighbors:
                    edata = self.G.get_edge_data(src, nbr) or self.G.get_edge_data(nbr, src) or {}
                    w = float(edata.get("weight", 0.5))
                    etype = edata.get("type", "")
                    boost = 1.3 if etype in priority_edge_types else 0.8
                    decay = 0.7  # synaptic decay per hop
                    spread_energy = act * w * boost * decay
                    activation[nbr] = max(activation.get(nbr, 0.0), spread_energy)

        # Rank by final activation energy
        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        results = []
        for nid, score in ranked[:top_k]:
            data = self.nodes_dict.get(nid, {})
            results.append({
                "id": nid,
                "text": data.get("statement", "") or data.get("name", ""),
                "node_type": data.get("node_type", ""),
                "activation": round(score, 4),
                "source": "brain_graph_spreading_activation"
            })
        return results


class VectorStoreSubstrate:
    """Baseline B: Flat Dense Vector Store (RAG Paradigm)."""
    def __init__(self, brain_data: dict, emb_index: EmbeddingIndex):
        self.nodes_dict = {n["id"]: n for n in brain_data["graph"]["nodes"]}
        self.emb_index = emb_index

    def retrieve(self, query: str, top_k: int = 5) -> list:
        q_vec = shared_embed(query)
        matches = self.emb_index.query(q_vec, threshold=0.3, top_k=top_k)
        results = []
        for nid, sim in matches:
            data = self.nodes_dict.get(nid, {})
            results.append({
                "id": nid,
                "text": data.get("statement", "") or data.get("name", ""),
                "node_type": data.get("node_type", ""),
                "similarity": round(float(sim), 4),
                "source": "vector_rag"
            })
        return results


class FlatFileSubstrate:
    """Baseline A: Flat-File Memory (AutoScientists EXPLORED.md / NOTES.md Paradigm)."""
    def __init__(self, brain_data: dict):
        self.chunks = []
        # Build linear file format mirroring AutoScientists NOTES.md & EXPLORED.md
        for n in brain_data["graph"]["nodes"]:
            ntype = n.get("node_type", "")
            stmt = n.get("statement", "") or n.get("name", "")
            emp = n.get("empirical_result", "")
            chunk_text = f"[{ntype.upper()}] {stmt}"
            if emp:
                chunk_text += f" (Result: {emp})"
            self.chunks.append({
                "id": n["id"],
                "text": chunk_text,
                "node_type": ntype
            })

    def retrieve(self, query: str, top_k: int = 5) -> list:
        # BM25-like term matching over flat chunks
        terms = set(re.findall(r'\w+', query.lower()))
        scored = []
        for chunk in self.chunks:
            chunk_terms = re.findall(r'\w+', chunk["text"].lower())
            if not chunk_terms:
                continue
            matches = sum(1 for t in terms if t in chunk_terms)
            if matches > 0:
                score = matches / (len(chunk_terms) ** 0.5)
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


# ── Benchmark Evaluation Battery ──────────────────────────────────────────────

BENCHMARK_TASKS = [
    {
        "task_id": "T1_analogy_quantum_dna",
        "category": "multi_hop_analogy",
        "query": "How do quantum error correction lattice mechanisms relate to biological low-entropy maintenance and DNA damage repair?",
        "target_keywords": ["surface code", "quantum", "entropy", "dna", "repair", "lattice"],
        "min_required_matches": 3
    },
    {
        "task_id": "T2_analogy_hartley_spiking",
        "category": "multi_hop_analogy",
        "query": "How does Hartley symbol distinguishability link to Shannon channel capacity and energy dissipation in spiking neural architectures?",
        "target_keywords": ["hartley", "shannon", "channel", "spiking", "energy", "dissipation"],
        "min_required_matches": 3
    },
    {
        "task_id": "T3_contradiction_rate_vs_temporal",
        "category": "contradiction_preservation",
        "query": "What is the conflict between the rate coding model and temporal coding in neuronal communication?",
        "target_keywords": ["rate coding", "temporal", "spike", "frequency", "timing"],
        "opposing_aspects": ["rate", "temporal"],
        "min_required_matches": 2
    },
    {
        "task_id": "T4_contradiction_soc_criticality",
        "category": "contradiction_preservation",
        "query": "Is self-organized criticality (SOC) a fundamental and necessary property of neural computation, or not?",
        "target_keywords": ["criticality", "soc", "fundamental", "property"],
        "opposing_aspects": ["fundamental", "not"],
        "min_required_matches": 2
    },
    {
        "task_id": "T5_empirical_bit_erasure_threshold",
        "category": "empirical_grounding",
        "query": "What empirical simulation was conducted to test energy thresholds for bit erasure in biological systems?",
        "target_keywords": ["erasure", "threshold", "energy", "simulation", "storage"],
        "require_empirical_node": True,
        "min_required_matches": 3
    },
    {
        "task_id": "T6_empirical_ferritin_tunneling",
        "category": "empirical_grounding",
        "query": "What experimental computational test demonstrated electron transport in ferritin tagged with quantum dots?",
        "target_keywords": ["ferritin", "quantum dot", "electron", "transport", "tunneling"],
        "require_empirical_node": True,
        "min_required_matches": 3
    }
]


def evaluate_retrieval(results: list, task: dict) -> dict:
    combined_text = " ".join(r["text"].lower() for r in results)
    types_found = set(r["node_type"] for r in results)

    # Keyword coverage
    matched_keywords = [kw for kw in task["target_keywords"] if kw.lower() in combined_text]
    keyword_recall = len(matched_keywords) / len(task["target_keywords"])

    success = len(matched_keywords) >= task["min_required_matches"]

    # Contradiction check
    opposing_retained = True
    if "opposing_aspects" in task:
        for opp in task["opposing_aspects"]:
            if opp.lower() not in combined_text:
                opposing_retained = False
        if not opposing_retained:
            success = False

    # Empirical check
    empirical_retained = True
    if task.get("require_empirical_node"):
        if "empirical" not in types_found:
            empirical_retained = False
            success = False

    return {
        "success": success,
        "matched_keywords": matched_keywords,
        "keyword_recall": round(keyword_recall, 3),
        "opposing_retained": opposing_retained,
        "empirical_retained": empirical_retained,
        "retrieved_count": len(results)
    }


def run_benchmark():
    print("=" * 70)
    print("AUTOSCIENTIST: REPRESENTATION BASELINE COMPARISON (PHASE 1)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading brain graph...")
    with open(BRAIN_PATH, 'r') as f:
        brain_data = json.load(f)

    print("Loading embedding index...")
    emb_index = EmbeddingIndex.load(INDEX_PATH)

    print("Initializing substrates:")
    print("  1. Condition C: Brain-as-Graph (NetworkX multi-relational)")
    brain_substrate = BrainGraphSubstrate(brain_data, emb_index)

    print("  2. Baseline B: Flat Vector Store (FAISS RAG)")
    vector_substrate = VectorStoreSubstrate(brain_data, emb_index)

    print("  3. Baseline A: Flat-File Memory (AutoScientists EXPLORED.md)")
    flat_file_substrate = FlatFileSubstrate(brain_data)

    substrates = {
        "BrainGraph": brain_substrate,
        "VectorRAG": vector_substrate,
        "FlatFile": flat_file_substrate
    }

    report = {"tasks": [], "aggregate": {}}

    for task in BENCHMARK_TASKS:
        print(f"\nEvaluating: [{task['category']}] {task['task_id']}")
        task_res = {"task_id": task["task_id"], "category": task["category"], "substrates": {}}

        for name, sub in substrates.items():
            results = sub.retrieve(task["query"], top_k=6)
            eval_res = evaluate_retrieval(results, task)
            task_res["substrates"][name] = eval_res
            status_str = "PASS" if eval_res["success"] else "FAIL"
            print(f"  [{name:12s}] {status_str} | Recall: {eval_res['keyword_recall']:.2f} | "
                  f"Opposing: {eval_res['opposing_retained']} | Empirical: {eval_res['empirical_retained']}")

        report["tasks"].append(task_res)

    # Compute aggregate scores
    agg = {}
    for name in substrates:
        total_tasks = len(BENCHMARK_TASKS)
        passed = sum(1 for t in report["tasks"] if t["substrates"][name]["success"])
        avg_recall = np.mean([t["substrates"][name]["keyword_recall"] for t in report["tasks"]])
        agg[name] = {
            "pass_rate": round(passed / total_tasks, 3),
            "passed": passed,
            "total": total_tasks,
            "mean_keyword_recall": round(float(avg_recall), 3)
        }

    report["aggregate"] = agg

    print("\n" + "=" * 70)
    print("AGGREGATE COMPARATIVE RESULTS:")
    print("=" * 70)
    for name, stats in agg.items():
        print(f"  {name:15s} | Pass Rate: {stats['pass_rate']*100:5.1f}% ({stats['passed']}/{stats['total']}) | "
              f"Mean Recall: {stats['mean_keyword_recall']:.3f}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed benchmark report saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_benchmark()
