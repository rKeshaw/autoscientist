#!/usr/bin/env python3
"""
test_representation_baselines_expanded.py

Expanded 24-Task Epistemic Benchmark comparing 3 Memory Substrates:
1. Condition C: Brain-as-Graph (NetworkX with Spreading Activation across typed synapses)
2. Baseline B: Flat Dense Vector Store (FAISS Inner Product / RAG paradigm)
3. Baseline A: Flat-File Memory (AutoScientists EXPLORED.md / NOTES.md linear chunking)

Test Categories (24 Tasks Total):
- Category 1: Multi-Hop Cross-Domain Analogies (6 Tasks)
- Category 2: Contradiction & Scientific Tension Preservation (6 Tasks)
- Category 3: Empirical Simulation & Code Grounding (6 Tasks)
- Category 4: Negative Knowledge, Failure Modes & Dead-End Avoidance (6 Tasks)
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "representation_comparison_expanded.json")


# ── Substrate Wrappers ────────────────────────────────────────────────────────

class BrainGraphSubstrate:
    """Condition C: Brain-as-Graph with neurobiological Spreading Activation."""
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

    def retrieve(self, query: str, top_k: int = 6, hop_depth: int = 2) -> list:
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

    def retrieve(self, query: str, top_k: int = 6) -> list:
        q_vec = shared_embed(query)
        matches = self.emb_index.query(q_vec, threshold=0.25, top_k=top_k)
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
        for n in brain_data["graph"]["nodes"]:
            ntype = n.get("node_type", "")
            stmt = n.get("statement", "") or n.get("name", "")
            emp = n.get("empirical_result", "")
            chunk_text = f"[{ntype.upper()}] {stmt}"
            if emp:
                chunk_text += f" (Empirical Result: {emp})"
            self.chunks.append({
                "id": n["id"],
                "text": chunk_text,
                "node_type": ntype
            })

    def retrieve(self, query: str, top_k: int = 6) -> list:
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


# ── Expanded 24-Task Benchmark Battery ────────────────────────────────────────

EXPANDED_BENCHMARK_TASKS = [
    # ── Category 1: Multi-Hop Cross-Domain Analogies (6 Tasks) ────────────────
    {
        "task_id": "T01_analogy_quantum_surface_dna",
        "category": "multi_hop_analogy",
        "query": "How do quantum error correction lattice mechanisms relate to biological low-entropy maintenance and DNA damage repair?",
        "target_keywords": ["surface code", "quantum", "entropy", "dna", "repair", "lattice"],
        "min_required_matches": 3
    },
    {
        "task_id": "T02_analogy_hartley_shannon_spiking",
        "category": "multi_hop_analogy",
        "query": "How does Hartley symbol distinguishability link to Shannon channel capacity and energy dissipation in spiking neural architectures?",
        "target_keywords": ["hartley", "shannon", "channel", "spiking", "energy", "dissipation"],
        "min_required_matches": 3
    },
    {
        "task_id": "T03_analogy_kinetic_proofreading_spiking",
        "category": "multi_hop_analogy",
        "query": "How can non-equilibrium kinetic proofreading mechanisms from biochemical copying be applied to error correction in spiking neuromorphic architectures?",
        "target_keywords": ["kinetic proofreading", "proofreading", "spiking", "neuromorphic", "error correction"],
        "min_required_matches": 3
    },
    {
        "task_id": "T04_analogy_brownian_ratchet_erasure",
        "category": "multi_hop_analogy",
        "query": "What role does the Feynman Brownian ratchet play in modeling directed molecular transport and thermodynamic bit erasure?",
        "target_keywords": ["brownian", "ratchet", "transport", "erasure", "thermodynamics"],
        "min_required_matches": 2
    },
    {
        "task_id": "T05_analogy_annealing_synaptic_pruning",
        "category": "multi_hop_analogy",
        "query": "How does simulated annealing in stochastic optimization map to synaptic pruning during REM sleep?",
        "target_keywords": ["simulated annealing", "annealing", "synaptic", "pruning", "rem", "sleep"],
        "min_required_matches": 3
    },
    {
        "task_id": "T06_analogy_ferritin_quantum_storage",
        "category": "multi_hop_analogy",
        "query": "How does electron transport in ferritin demonstrate biological quantum properties for energy-efficient information storage?",
        "target_keywords": ["ferritin", "electron", "quantum", "storage", "transport"],
        "min_required_matches": 3
    },

    # ── Category 2: Contradiction & Scientific Tension Preservation (6 Tasks) ─
    {
        "task_id": "T07_contradiction_rate_vs_temporal_coding",
        "category": "contradiction_preservation",
        "query": "What is the unresolved conflict between the rate coding model and temporal coding in neuronal communication?",
        "target_keywords": ["rate coding", "temporal", "spike", "frequency", "timing"],
        "opposing_aspects": ["rate", "temporal"],
        "min_required_matches": 2
    },
    {
        "task_id": "T08_contradiction_soc_criticality_necessity",
        "category": "contradiction_preservation",
        "query": "Is self-organized criticality (SOC) a fundamental physical necessity for neural computation, or can neural systems function without criticality?",
        "target_keywords": ["criticality", "soc", "fundamental", "property"],
        "opposing_aspects": ["fundamental", "not"],
        "min_required_matches": 2
    },
    {
        "task_id": "T09_contradiction_landauer_angular_momentum",
        "category": "contradiction_preservation",
        "query": "Can conserved quantities like angular momentum permit bit erasure without energy dissipation, challenging Landauer's bound?",
        "target_keywords": ["landauer", "angular momentum", "erasure", "dissipation", "entropy"],
        "opposing_aspects": ["landauer", "angular"],
        "min_required_matches": 2
    },
    {
        "task_id": "T10_contradiction_neuromodulatory_vs_structural",
        "category": "contradiction_preservation",
        "query": "Do adaptive changes in neural networks occur solely through neuromodulatory influences or through autonomous structural synaptic growth?",
        "target_keywords": ["neuromodulatory", "adaptive", "structural", "synaptic"],
        "opposing_aspects": ["neuromodulatory", "structural"],
        "min_required_matches": 2
    },
    {
        "task_id": "T11_contradiction_classical_prob_vs_quantum_cognition",
        "category": "contradiction_preservation",
        "query": "Why does classical probability theory fail to explain human psychological fallacies like the conjunction fallacy, and how does quantum probability contrast with it?",
        "target_keywords": ["classical", "probability", "quantum", "conjunction", "fallacy"],
        "opposing_aspects": ["classical", "quantum"],
        "min_required_matches": 2
    },
    {
        "task_id": "T12_contradiction_deterministic_vs_stochastic_biology",
        "category": "contradiction_preservation",
        "query": "What is the tension between deterministic ODE models and stochastic master equations in low-copy gene regulatory networks?",
        "target_keywords": ["deterministic", "stochastic", "gene", "regulatory", "ode"],
        "opposing_aspects": ["deterministic", "stochastic"],
        "min_required_matches": 2
    },

    # ── Category 3: Empirical Grounding & Code/Parameter Retrieval (6 Tasks) ───
    {
        "task_id": "T13_empirical_bit_erasure_energy_sweep",
        "category": "empirical_grounding",
        "query": "What empirical simulation was conducted to test energy thresholds for bit erasure in biological systems, and what was the verdict?",
        "target_keywords": ["erasure", "threshold", "energy", "simulation", "storage"],
        "require_empirical_node": True,
        "min_required_matches": 3
    },
    {
        "task_id": "T14_empirical_ferritin_quantum_dots",
        "category": "empirical_grounding",
        "query": "What experimental computational simulation tested electron transport in ferritin tagged with quantum dots, and was a plot saved?",
        "target_keywords": ["ferritin", "quantum dot", "electron", "transport", "tunneling"],
        "require_empirical_node": True,
        "min_required_matches": 3
    },
    {
        "task_id": "T15_empirical_quantum_cognition_conjunction",
        "category": "empirical_grounding",
        "query": "What computational test was run on quantum probability modeling of human probability judgment and the conjunction fallacy?",
        "target_keywords": ["conjunction", "quantum", "probability", "fallacy", "simulation"],
        "require_empirical_node": True,
        "min_required_matches": 3
    },
    {
        "task_id": "T16_empirical_membrane_computing_model",
        "category": "empirical_grounding",
        "query": "What computational test evaluated membrane computing cellular models?",
        "target_keywords": ["membrane computing", "cellular", "model", "test"],
        "require_empirical_node": True,
        "min_required_matches": 2
    },
    {
        "task_id": "T17_empirical_sure_thing_principle",
        "category": "empirical_grounding",
        "query": "What computational simulation examined the sure-thing principle in decision-making?",
        "target_keywords": ["sure-thing", "decision", "principle", "simulation"],
        "require_empirical_node": True,
        "min_required_matches": 2
    },
    {
        "task_id": "T18_empirical_classical_probability_psychology",
        "category": "empirical_grounding",
        "query": "What computational test evaluated classical probability theory against human inference observations?",
        "target_keywords": ["classical probability", "inference", "psychology", "computational test"],
        "require_empirical_node": True,
        "min_required_matches": 2
    },

    # ── Category 4: Negative Knowledge, Failure Diagnosis & Dead-Ends (6 Tasks)
    {
        "task_id": "T19_negative_inhomogeneous_array_shape_error",
        "category": "negative_knowledge",
        "query": "What execution error occurred when trying to pass array parameters to solve_ivp in neuron simulation, leading to inhomogeneous shape ValueError?",
        "target_keywords": ["solve_ivp", "inhomogeneous", "shape", "valueerror", "neuron"],
        "min_required_matches": 2
    },
    {
        "task_id": "T20_negative_brian2_numpy2_incompatibility",
        "category": "negative_knowledge",
        "query": "What runtime AttributeError occurred when importing Brian2 under NumPy 2.x due to the removal of np.ndarray.ptp?",
        "target_keywords": ["brian2", "numpy", "ptp", "attributeerror"],
        "min_required_matches": 2
    },
    {
        "task_id": "T21_negative_syntax_error_docstring_repair",
        "category": "negative_knowledge",
        "query": "What syntax error occurred during the quantum dot ferritin code generation attempt 1, and how was it corrected in attempt 2?",
        "target_keywords": ["syntaxerror", "triple-quoted", "string", "attempt", "ferritin"],
        "min_required_matches": 2
    },
    {
        "task_id": "T22_negative_system2_redundant_synthesis_rejection",
        "category": "negative_knowledge",
        "query": "Why did System 2 Critic reject the synthesis claim about investigating energy thresholds for bit erasure?",
        "target_keywords": ["redundant", "reject", "critic", "bit erasure", "threshold"],
        "min_required_matches": 2
    },
    {
        "task_id": "T23_negative_procedural_policy_penalty",
        "category": "negative_knowledge",
        "query": "How was the procedural thinking policy penalized following the generation of redundant sub-questions?",
        "target_keywords": ["policy", "reductive", "rew", "val", "thinking"],
        "min_required_matches": 2
    },
    {
        "task_id": "T24_negative_stale_insight_pruning",
        "category": "negative_knowledge",
        "query": "What mechanism prunes stale candidate pairs and decays unreinforced edges during consolidation?",
        "target_keywords": ["pruned", "stale", "decay", "consolidation", "pairs"],
        "min_required_matches": 2
    }
]


def evaluate_task_retrieval(results: list, task: dict) -> dict:
    combined_text = " ".join(r["text"].lower() for r in results)
    types_found = set(r["node_type"] for r in results)

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

    # Empirical node check
    empirical_retained = True
    if task.get("require_empirical_node"):
        if "empirical" not in types_found:
            empirical_retained = False
            success = False

    # Reciprocal rank (first item containing at least half target keywords)
    mrr = 0.0
    for rank, r in enumerate(results, start=1):
        r_text = r["text"].lower()
        r_matches = sum(1 for kw in task["target_keywords"] if kw.lower() in r_text)
        if r_matches >= max(1, task["min_required_matches"] // 2):
            mrr = 1.0 / rank
            break

    return {
        "success": success,
        "matched_keywords": matched_keywords,
        "keyword_recall": round(keyword_recall, 3),
        "mrr": round(mrr, 3),
        "opposing_retained": opposing_retained,
        "empirical_retained": empirical_retained,
        "retrieved_count": len(results)
    }


def run_expanded_benchmark():
    print("=" * 80)
    print("AUTOSCIENTIST: EXPANDED 24-TASK EPISTEMIC BENCHMARK")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(BRAIN_PATH, 'r') as f:
        brain_data = json.load(f)

    emb_index = EmbeddingIndex.load(INDEX_PATH)

    print("Initializing Substrates:")
    print("  1. Condition C: Brain-as-Graph (NetworkX multi-relational + Spreading Activation)")
    brain_sub = BrainGraphSubstrate(brain_data, emb_index)

    print("  2. Baseline B: Flat Vector Store (FAISS Inner Product / RAG)")
    vec_sub = VectorStoreSubstrate(brain_data, emb_index)

    print("  3. Baseline A: Flat-File Memory (AutoScientists EXPLORED.md / NOTES.md)")
    flat_sub = FlatFileSubstrate(brain_data)

    substrates = {
        "BrainGraph": brain_sub,
        "VectorRAG": vec_sub,
        "FlatFile": flat_sub
    }

    report = {"tasks": [], "by_category": {}, "aggregate": {}}

    categories = ["multi_hop_analogy", "contradiction_preservation", "empirical_grounding", "negative_knowledge"]

    for idx, task in enumerate(EXPANDED_BENCHMARK_TASKS, start=1):
        cat = task["category"]
        tid = task["task_id"]
        print(f"\nTask {idx:02d}/24 [{cat}]: {tid}")

        task_record = {"task_id": tid, "category": cat, "substrates": {}}

        for name, sub in substrates.items():
            results = sub.retrieve(task["query"], top_k=6)
            eval_res = evaluate_task_retrieval(results, task)
            task_record["substrates"][name] = eval_res
            status = "PASS" if eval_res["success"] else "FAIL"
            print(f"  [{name:10s}] {status:4s} | Recall: {eval_res['keyword_recall']:.2f} | "
                  f"MRR: {eval_res['mrr']:.2f} | Opposing: {eval_res['opposing_retained']} | Emp: {eval_res['empirical_retained']}")

        report["tasks"].append(task_record)

    # Category breakdowns
    for cat in categories:
        cat_tasks = [t for t in report["tasks"] if t["category"] == cat]
        report["by_category"][cat] = {}
        for name in substrates:
            n_cat = len(cat_tasks)
            passed = sum(1 for t in cat_tasks if t["substrates"][name]["success"])
            mean_rec = np.mean([t["substrates"][name]["keyword_recall"] for t in cat_tasks])
            mean_mrr = np.mean([t["substrates"][name]["mrr"] for t in cat_tasks])
            report["by_category"][cat][name] = {
                "pass_rate": round(passed / n_cat, 3),
                "passed": passed,
                "total": n_cat,
                "mean_recall": round(float(mean_rec), 3),
                "mean_mrr": round(float(mean_mrr), 3)
            }

    # Aggregate scores
    for name in substrates:
        total = len(EXPANDED_BENCHMARK_TASKS)
        passed = sum(1 for t in report["tasks"] if t["substrates"][name]["success"])
        mean_rec = np.mean([t["substrates"][name]["keyword_recall"] for t in report["tasks"]])
        mean_mrr = np.mean([t["substrates"][name]["mrr"] for t in report["tasks"]])
        report["aggregate"][name] = {
            "pass_rate": round(passed / total, 3),
            "passed": passed,
            "total": total,
            "mean_recall": round(float(mean_rec), 3),
            "mean_mrr": round(float(mean_mrr), 3)
        }

    print("\n" + "=" * 80)
    print("EXPANDED BENCHMARK SUMMARY (24 TASKS ACROSS 4 CATEGORIES):")
    print("=" * 80)
    print(f"{'Substrate':15s} | {'Overall Pass Rate':20s} | {'Mean Recall':12s} | {'Mean MRR':10s}")
    print("-" * 65)
    for name, stats in report["aggregate"].items():
        print(f"{name:15s} | {stats['pass_rate']*100:5.1f}% ({stats['passed']}/{stats['total']})       | "
              f"{stats['mean_recall']:.3f}        | {stats['mean_mrr']:.3f}")

    print("\nCATEGORY BREAKDOWN:")
    for cat in categories:
        print(f"\n[{cat.upper()}]:")
        for name in substrates:
            cs = report["by_category"][cat][name]
            print(f"  {name:12s} | Pass: {cs['pass_rate']*100:5.1f}% ({cs['passed']}/{cs['total']}) | "
                  f"Recall: {cs['mean_recall']:.3f} | MRR: {cs['mean_mrr']:.3f}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved expanded benchmark results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_expanded_benchmark()
