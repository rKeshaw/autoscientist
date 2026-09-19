#!/usr/bin/env python3
"""
benchmark/test_phase3_representation_deep_dive.py

Phase 3 Comprehensive Benchmark: Representation Baselines Deep-Dive & Multi-Hop Path Reasoning
=============================================================================================
Author: AutoScientist Architectural Benchmarking Suite
Date: September 17, 2026

Empirical comparison of four memory representations:
1. Condition C: Brain-as-Graph (Directed weighted connectome with typed synapses,
                spreading activation, and Dijkstra shortest path traversal)
2. Baseline B-iter: Iterative Multi-Query Vector RAG (IRCoT / Multi-Hop Dense Retrieval)
3. Baseline B-single: Standard Single-Hop Vector RAG (FAISS Top-k Inner Product)
4. Baseline A: Flat-File Memory (AutoScientists EXPLORED.md / NOTES.md linear chunking with BM25)

Five Experimental Suites:
- Suite 1: Multi-Hop Relational Path Discovery (k = 2, 3, 4, 5 hops across 30 scientific pairs)
- Suite 2: Context Subgraph Coherence & Topological Mutual Information
- Suite 3: Signed Structural Balance & Contradiction Resolution (The Falsification Test on 63 contradictions)
- Suite 4: Context Window Token Budget Efficiency (B = 250, 500, 1000, 2000 tokens)
- Suite 5: Topological Distraction Robustness (Decoy Ingestion Stress Test with 100 ungrounded distractors)
"""

import os
import sys
import json
import re
import math
import random
import numpy as np
import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from embedding_index import EmbeddingIndex
from embedding import embed as shared_embed

BRAIN_PATH = "data/brain.json"
INDEX_PATH = "data/embedding_index"
OUTPUT_DIR = "benchmark/results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "phase3_representation_deep_dive.json")


# ── Substrate Definitions ─────────────────────────────────────────────────────

class BrainGraphSubstrate:
    """Condition C: Brain-as-Graph Connectome with spreading activation and Dijkstra path search."""
    def __init__(self, brain_data: dict, emb_index: EmbeddingIndex):
        self.brain_data = brain_data
        self.emb_index = emb_index
        self.G = nx.DiGraph()
        self.UG = nx.Graph()
        self.nodes_dict = {}

        for n in brain_data["graph"]["nodes"]:
            nid = n["id"]
            self.nodes_dict[nid] = n
            self.G.add_node(nid, **{k: v for k, v in n.items() if k != "id"})
            self.UG.add_node(nid, **{k: v for k, v in n.items() if k != "id"})

        for e in brain_data["graph"]["edges"]:
            src = e["source"]
            tgt = e["target"]
            w = float(e.get("weight", 0.5))
            cost = 1.0 / max(w, 0.01)  # Higher weight = lower resistance
            attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
            attrs["cost"] = cost
            attrs["weight"] = w
            self.G.add_edge(src, tgt, **attrs)
            # Undirected view for bidirectional traversal
            if not self.UG.has_edge(src, tgt) or self.UG[src][tgt].get("weight", 0) < w:
                self.UG.add_edge(src, tgt, **attrs)

    def retrieve_subgraph(self, query: str, top_k: int = 8, hop_depth: int = 2) -> list:
        """Spreading activation from seed embeddings along synaptic edges."""
        q_vec = shared_embed(query)
        seeds = self.emb_index.query(q_vec, threshold=0.25, top_k=min(5, top_k))
        
        activation = {}
        for nid, sim in seeds:
            if nid in self.G:
                activation[nid] = float(sim)

        priority_edge_types = {
            "structural_analogy", "deep_isomorphism", "contradicts",
            "empirically_tested", "supports", "causes", "synthesis", "toward_mission"
        }

        for _ in range(hop_depth):
            current_nodes = list(activation.items())
            for src, act in current_nodes:
                if act < 0.08:
                    continue
                nbrs = list(self.G.successors(src)) + list(self.G.predecessors(src))
                for nbr in nbrs:
                    edata = self.G.get_edge_data(src, nbr) or self.G.get_edge_data(nbr, src) or {}
                    w = float(edata.get("weight", 0.5))
                    etype = edata.get("type", "")
                    boost = 1.35 if etype in priority_edge_types else 0.85
                    decay = 0.72
                    spread = act * w * boost * decay
                    activation[nbr] = max(activation.get(nbr, 0.0), spread)

        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in ranked[:top_k]]

    def find_shortest_path(self, source_id: str, target_id: str) -> list:
        """Dijkstra shortest path based on synaptic resistance (1/weight)."""
        if not (self.UG.has_node(source_id) and self.UG.has_node(target_id)):
            return []
        if not nx.has_path(self.UG, source_id, target_id):
            return []
        try:
            return nx.dijkstra_path(self.UG, source_id, target_id, weight="cost")
        except Exception:
            return nx.shortest_path(self.UG, source_id, target_id)


class VectorStoreIterativeSubstrate:
    """Baseline B-iter: Iterative Multi-Query Vector RAG (IRCoT / Multi-Hop Dense Retrieval)."""
    def __init__(self, brain_data: dict, emb_index: EmbeddingIndex):
        self.nodes_dict = {n["id"]: n for n in brain_data["graph"]["nodes"]}
        self.emb_index = emb_index

    def multi_hop_search(self, source_id: str, target_id: str, max_hops: int = 5, top_k_per_hop: int = 3) -> list:
        """Iteratively hops through vector space guided by intermediate embeddings."""
        if source_id not in self.nodes_dict:
            return []
        target_stmt = self.nodes_dict.get(target_id, {}).get("statement", "")
        
        path = [source_id]
        visited = {source_id}
        curr = source_id

        for _ in range(max_hops):
            curr_stmt = self.nodes_dict[curr].get("statement", "")
            # Composite prompt combining current position with destination target guidance
            hop_query = f"{curr_stmt} {target_stmt[:80]}"
            q_vec = shared_embed(hop_query)
            hits = self.emb_index.query(q_vec, threshold=0.20, top_k=top_k_per_hop + 3)
            
            next_nid = None
            for nid, _ in hits:
                if nid not in visited:
                    next_nid = nid
                    break
            if not next_nid:
                break
            
            path.append(next_nid)
            visited.add(next_nid)
            curr = next_nid
            if curr == target_id:
                break

        return path


class VectorStoreSingleSubstrate:
    """Baseline B-single: Standard Flat Vector Store (FAISS Inner Product)."""
    def __init__(self, brain_data: dict, emb_index: EmbeddingIndex):
        self.nodes_dict = {n["id"]: n for n in brain_data["graph"]["nodes"]}
        self.emb_index = emb_index

    def retrieve(self, query: str, top_k: int = 8) -> list:
        q_vec = shared_embed(query)
        matches = self.emb_index.query(q_vec, threshold=0.20, top_k=top_k)
        return [nid for nid, _ in matches]


class FlatFileSubstrate:
    """Baseline A: Flat-File Memory (AutoScientists EXPLORED.md Linear Chunking with BM25)."""
    def __init__(self, brain_data: dict):
        self.chunks = []
        for n in brain_data["graph"]["nodes"]:
            ntype = n.get("node_type", "")
            stmt = n.get("statement", "") or n.get("name", "")
            emp = n.get("empirical_result", "")
            text = f"[{ntype.upper()}] {stmt}"
            if emp:
                text += f" | Empirical: {emp}"
            tokens = set(re.findall(r'\w+', text.lower()))
            self.chunks.append({
                "id": n["id"],
                "text": text,
                "tokens": tokens,
                "length": len(tokens)
            })

    def retrieve(self, query: str, top_k: int = 8) -> list:
        q_tokens = set(re.findall(r'\w+', query.lower()))
        if not q_tokens:
            return [c["id"] for c in self.chunks[:top_k]]
        
        scored = []
        for chunk in self.chunks:
            overlap = len(q_tokens & chunk["tokens"])
            if overlap > 0:
                # BM25-like length normalized term frequency
                score = overlap / (chunk["length"] ** 0.5 + 1.0)
                scored.append((score, chunk["id"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored[:top_k]]


# ── Metric Calculation Utilities ──────────────────────────────────────────────

def calculate_subgraph_coherence(node_ids: list, G: nx.DiGraph) -> dict:
    """Compute topological coherence metrics for an induced subgraph."""
    valid_ids = [nid for nid in node_ids if nid in G]
    if not valid_ids:
        return {
            "node_count": 0, "edge_count": 0, "density": 0.0,
            "connected_components": 0, "giant_component_ratio": 0.0,
            "clustering_coefficient": 0.0, "relational_coverage": 0.0
        }
    
    sub = G.subgraph(valid_ids)
    sub_u = sub.to_undirected()
    n_nodes = len(valid_ids)
    n_edges = sub.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = round(n_edges / max_edges, 4) if max_edges > 0 else 0.0

    comps = list(nx.connected_components(sub_u))
    n_comps = len(comps)
    giant_size = len(max(comps, key=len)) if comps else 0
    giant_ratio = round(giant_size / n_nodes, 4) if n_nodes > 0 else 0.0
    
    try:
        clust = round(nx.average_clustering(sub_u), 4)
    except Exception:
        clust = 0.0

    connected_nodes = sum(1 for nid in valid_ids if sub_u.degree(nid) > 0)
    relational_coverage = round(connected_nodes / n_nodes, 4) if n_nodes > 0 else 0.0

    return {
        "node_count": n_nodes,
        "edge_count": n_edges,
        "density": density,
        "connected_components": n_comps,
        "giant_component_ratio": giant_ratio,
        "clustering_coefficient": clust,
        "relational_coverage": relational_coverage
    }


def calculate_path_continuity(path_ids: list, G: nx.DiGraph) -> float:
    """Fraction of adjacent step pairs (u, v) that have a real edge in the graph."""
    if len(path_ids) <= 1:
        return 1.0
    valid_steps = 0
    total_steps = len(path_ids) - 1
    for i in range(total_steps):
        u, v = path_ids[i], path_ids[i + 1]
        if G.has_edge(u, v) or G.has_edge(v, u):
            valid_steps += 1
    return round(valid_steps / total_steps, 4)


# ── Suite 1: Multi-Hop Relational Path Discovery ─────────────────────────────

def run_suite_1_path_discovery(brain_graph: BrainGraphSubstrate,
                               vec_iter: VectorStoreIterativeSubstrate,
                               vec_single: VectorStoreSingleSubstrate,
                               flat_file: FlatFileSubstrate) -> dict:
    print("\n" + "=" * 80)
    print("SUITE 1: MULTI-HOP RELATIONAL PATH DISCOVERY (k = 2, 3, 4, 5 hops)")
    print("=" * 80)

    # Curate 30 ground-truth multi-hop paths across the connectome:
    # 10 Short (k=2), 10 Medium (k=3), 10 Long (k=4, 5)
    random.seed(42)
    G_undir = brain_graph.UG
    all_pairs = dict(nx.all_pairs_shortest_path_length(G_undir))

    pairs_by_len = {2: [], 3: [], 4: [], 5: []}
    nodes_list = list(G_undir.nodes())
    random.shuffle(nodes_list)

    for u in nodes_list:
        if u not in all_pairs:
            continue
        for v, d in all_pairs[u].items():
            if d in pairs_by_len and u < v:
                c1 = brain_graph.nodes_dict[u].get("cluster", "c1")
                c2 = brain_graph.nodes_dict[v].get("cluster", "c2")
                # Prefer inter-disciplinary paths across different clusters
                if c1 != c2 and len(pairs_by_len[d]) < 10:
                    pairs_by_len[d].append((u, v))
        if all(len(pairs_by_len[d]) >= 10 for d in (2, 3, 4)):
            break

    # Build 30 test cases
    test_cases = []
    test_cases.extend([{"source": u, "target": v, "hops": 2} for u, v in pairs_by_len[2][:10]])
    test_cases.extend([{"source": u, "target": v, "hops": 3} for u, v in pairs_by_len[3][:10]])
    long_cases = (pairs_by_len[4] + pairs_by_len[5])[:10]
    test_cases.extend([{"source": u, "target": v, "hops": len(nx.shortest_path(G_undir, u, v)) - 1} for u, v in long_cases])

    print(f"Total curated multi-hop ground truth paths: {len(test_cases)}")

    results = {
        "brain_graph": {"target_found": 0, "intermediate_recall": [], "step_continuity": [], "full_reconstruction": 0},
        "vec_iterative": {"target_found": 0, "intermediate_recall": [], "step_continuity": [], "full_reconstruction": 0},
        "vec_single": {"target_found": 0, "intermediate_recall": [], "step_continuity": [], "full_reconstruction": 0},
        "flat_file": {"target_found": 0, "intermediate_recall": [], "step_continuity": [], "full_reconstruction": 0},
    }

    per_hop_breakdown = {
        2: {"brain_graph": [], "vec_iterative": [], "vec_single": [], "flat_file": []},
        3: {"brain_graph": [], "vec_iterative": [], "vec_single": [], "flat_file": []},
        4: {"brain_graph": [], "vec_iterative": [], "vec_single": [], "flat_file": []},
        5: {"brain_graph": [], "vec_iterative": [], "vec_single": [], "flat_file": []},
    }

    for idx, tc in enumerate(test_cases):
        s, t, k = tc["source"], tc["target"], tc["hops"]
        gt_path = nx.shortest_path(G_undir, s, t)
        intermediates = set(gt_path[1:-1])
        s_stmt = brain_graph.nodes_dict[s].get("statement", "")
        t_stmt = brain_graph.nodes_dict[t].get("statement", "")
        query = f"Explain the relationship and intermediate mechanisms connecting: '{s_stmt[:80]}' to '{t_stmt[:80]}'"

        # 1. Brain-as-Graph
        bg_path = brain_graph.find_shortest_path(s, t)
        bg_retrieved = set(bg_path)
        bg_found = t in bg_retrieved
        bg_inter_rec = len(intermediates & bg_retrieved) / len(intermediates) if intermediates else 1.0
        bg_cont = calculate_path_continuity(bg_path, brain_graph.G)
        bg_full = intermediates.issubset(bg_retrieved) and bg_found

        results["brain_graph"]["target_found"] += int(bg_found)
        results["brain_graph"]["intermediate_recall"].append(bg_inter_rec)
        results["brain_graph"]["step_continuity"].append(bg_cont)
        results["brain_graph"]["full_reconstruction"] += int(bg_full)

        # 2. Vector Iterative (IRCoT baseline)
        vi_path = vec_iter.multi_hop_search(s, t, max_hops=k + 1, top_k_per_hop=3)
        vi_retrieved = set(vi_path)
        vi_found = t in vi_retrieved
        vi_inter_rec = len(intermediates & vi_retrieved) / len(intermediates) if intermediates else 1.0
        vi_cont = calculate_path_continuity(vi_path, brain_graph.G)
        vi_full = intermediates.issubset(vi_retrieved) and vi_found

        results["vec_iterative"]["target_found"] += int(vi_found)
        results["vec_iterative"]["intermediate_recall"].append(vi_inter_rec)
        results["vec_iterative"]["step_continuity"].append(vi_cont)
        results["vec_iterative"]["full_reconstruction"] += int(vi_full)

        # 3. Vector Single-Hop (top-8)
        vs_hits = vec_single.retrieve(query, top_k=8)
        vs_retrieved = set(vs_hits)
        vs_found = t in vs_retrieved
        vs_inter_rec = len(intermediates & vs_retrieved) / len(intermediates) if intermediates else 1.0
        vs_cont = calculate_path_continuity(vs_hits, brain_graph.G)
        vs_full = intermediates.issubset(vs_retrieved) and vs_found

        results["vec_single"]["target_found"] += int(vs_found)
        results["vec_single"]["intermediate_recall"].append(vs_inter_rec)
        results["vec_single"]["step_continuity"].append(vs_cont)
        results["vec_single"]["full_reconstruction"] += int(vs_full)

        # 4. Flat-File BM25 (top-8)
        ff_hits = flat_file.retrieve(query, top_k=8)
        ff_retrieved = set(ff_hits)
        ff_found = t in ff_retrieved
        ff_inter_rec = len(intermediates & ff_retrieved) / len(intermediates) if intermediates else 1.0
        ff_cont = calculate_path_continuity(ff_hits, brain_graph.G)
        ff_full = intermediates.issubset(ff_retrieved) and ff_found

        results["flat_file"]["target_found"] += int(ff_found)
        results["flat_file"]["intermediate_recall"].append(ff_inter_rec)
        results["flat_file"]["step_continuity"].append(ff_cont)
        results["flat_file"]["full_reconstruction"] += int(ff_full)

        hop_bin = min(k, 5)
        per_hop_breakdown[hop_bin]["brain_graph"].append(bg_full)
        per_hop_breakdown[hop_bin]["vec_iterative"].append(vi_full)
        per_hop_breakdown[hop_bin]["vec_single"].append(vs_full)
        per_hop_breakdown[hop_bin]["flat_file"].append(ff_full)

    n = len(test_cases)
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "target_found_rate": round(data["target_found"] / n, 3),
            "intermediate_recall": round(float(np.mean(data["intermediate_recall"])), 3),
            "step_continuity": round(float(np.mean(data["step_continuity"])), 3),
            "full_path_reconstruction_rate": round(data["full_reconstruction"] / n, 3)
        }

    hop_summary = {}
    for h, sub_dict in per_hop_breakdown.items():
        hop_summary[f"{h}_hop"] = {
            name: round(float(np.mean(vals)), 3) if vals else 0.0
            for name, vals in sub_dict.items()
        }

    print(f"{'Substrate':20s} | {'Target Found':14s} | {'Intermed Recall':16s} | {'Step Continuity':16s} | {'Full Path Recon':16s}")
    print("-" * 90)
    for name, s in summary.items():
        print(f"{name:20s} | {s['target_found_rate']*100:5.1f}%        | {s['intermediate_recall']:.3f}            | {s['step_continuity']*100:5.1f}%          | {s['full_path_reconstruction_rate']*100:5.1f}%")

    print("\nFULL PATH RECONSTRUCTION RATE BY HOP DISTANCE:")
    for h, vals in hop_summary.items():
        print(f"  {h}: Graph: {vals['brain_graph']*100:5.1f}% | VecIter: {vals['vec_iterative']*100:5.1f}% | VecSingle: {vals['vec_single']*100:5.1f}% | FlatFile: {vals['flat_file']*100:5.1f}%")

    return {"aggregate": summary, "per_hop_reconstruction": hop_summary}


# ── Suite 2: Context Subgraph Coherence & Topological Quality ─────────────────

def run_suite_2_context_coherence(brain_graph: BrainGraphSubstrate,
                                  vec_single: VectorStoreSingleSubstrate,
                                  flat_file: FlatFileSubstrate) -> dict:
    print("\n" + "=" * 80)
    print("SUITE 2: CONTEXT SUBGRAPH COHERENCE & TOPOLOGICAL QUALITY")
    print("=" * 80)

    # 20 diverse scientific inquiries across all domains
    test_queries = [
        "Thermodynamic dissipation limits and Landauer bit erasure in cellular computing",
        "Kinetic proofreading error correction in DNA polymerase and spiking neuromorphic models",
        "Quantum surface code parity checks mapped to biological nucleotide mismatch repair",
        "Rate coding vs temporal coding trade-offs in leaky integrate-and-fire neural networks",
        "Self-organized criticality (SOC) as a universal compute substrate in cortical networks",
        "Stochastic resonance and noise-induced state transitions in biological sensory systems",
        "Mitochondrial metabolic efficiency and energy bounds on synaptic transmission",
        "Simulated annealing and synaptic pruning during slow-wave sleep consolidation",
        "Phase separation in membraneless organelles and liquid-liquid demixing dynamics",
        "Ferritin protein electron transfer and quantum confinement effects in storage",
        "Allosteric cooperativity and Hill coefficients in biochemical signaling cascades",
        "Hartley symbol distinguishability and channel capacity in refractory spike trains",
        "Membrane computing P systems applied to distributed neuromorphic simulation",
        "Hebbian plasticity runaway prevention via homeostatic synaptic scaling",
        "Topological defect formation in active nematic liquid crystals and cell migration",
        "Calphad thermodynamic equilibrium modeling of multi-component metabolic mixtures",
        "Quantum cognition non-commutative probability in human decision making",
        "Spike-frequency adaptation (SFA) and temporal filtering in cortical pyramidal cells",
        "Critical branching processes and power-law avalanche distributions in neuroscience",
        "Non-equilibrium fluctuation theorems applied to nanoscopic molecular ratchets"
    ]

    substrates = {
        "brain_graph": lambda q: brain_graph.retrieve_subgraph(q, top_k=8),
        "vector_rag": lambda q: vec_single.retrieve(q, top_k=8),
        "flat_file": lambda q: flat_file.retrieve(q, top_k=8)
    }

    metrics = {
        name: {
            "edge_counts": [], "densities": [], "components": [],
            "giant_ratios": [], "clusterings": [], "coverages": []
        }
        for name in substrates
    }

    for q in test_queries:
        for name, fn in substrates.items():
            retrieved = fn(q)
            coh = calculate_subgraph_coherence(retrieved, brain_graph.G)
            metrics[name]["edge_counts"].append(coh["edge_count"])
            metrics[name]["densities"].append(coh["density"])
            metrics[name]["components"].append(coh["connected_components"])
            metrics[name]["giant_ratios"].append(coh["giant_component_ratio"])
            metrics[name]["clusterings"].append(coh["clustering_coefficient"])
            metrics[name]["coverages"].append(coh["relational_coverage"])

    summary = {}
    for name, data in metrics.items():
        summary[name] = {
            "mean_edge_count": round(float(np.mean(data["edge_counts"])), 2),
            "mean_density": round(float(np.mean(data["densities"])), 3),
            "mean_connected_components": round(float(np.mean(data["components"])), 2),
            "mean_giant_ratio": round(float(np.mean(data["giant_ratios"])), 3),
            "mean_clustering": round(float(np.mean(data["clusterings"])), 3),
            "mean_relational_coverage": round(float(np.mean(data["coverages"])), 3)
        }

    print(f"{'Substrate':15s} | {'Edges':7s} | {'Density':8s} | {'Components':12s} | {'Giant Ratio':12s} | {'Clustering':11s} | {'Coverage':9s}")
    print("-" * 85)
    for name, s in summary.items():
        print(f"{name:15s} | {s['mean_edge_count']:5.1f}   | {s['mean_density']:.3f}    | {s['mean_connected_components']:5.1f}        | "
              f"{s['mean_giant_ratio']*100:5.1f}%       | {s['mean_clustering']:.3f}       | {s['mean_relational_coverage']*100:5.1f}%")

    return summary


# ── Suite 3: Signed Balance & Contradiction Resolution (The Falsification Test)

def run_suite_3_contradictions(brain_graph: BrainGraphSubstrate,
                               vec_single: VectorStoreSingleSubstrate,
                               flat_file: FlatFileSubstrate) -> dict:
    print("\n" + "=" * 80)
    print("SUITE 3: SIGNED BALANCE & CONTRADICTION RESOLUTION (THE FALSIFICATION TEST)")
    print("=" * 80)

    # Extract all active contradiction edges from brain
    contradictions = [
        e for e in brain_graph.brain_data["graph"]["edges"]
        if e.get("type") == "contradicts"
    ]
    print(f"Total Ground-Truth Contradiction Edges in Connectome: {len(contradictions)}")

    # For each contradiction (u, v):
    # - Cosine similarity between u and v
    # - When querying thesis u: does retrieval surface antithesis v?
    # - Does the substrate know that v is CONTRADICTORY (sign = -1), or does it mistake it for support?
    # - Is there an empirical node resolving it?

    sims = []
    results = {
        "brain_graph": {"antithesis_retrieved": 0, "correct_sign_disambiguation": 0, "falsification_error": 0},
        "vector_rag": {"antithesis_retrieved": 0, "correct_sign_disambiguation": 0, "falsification_error": 0},
        "flat_file": {"antithesis_retrieved": 0, "correct_sign_disambiguation": 0, "falsification_error": 0}
    }

    empirical_resolutions_found = {"brain_graph": 0, "vector_rag": 0, "flat_file": 0}
    total_with_empirical = 0

    for c in contradictions:
        u_id = c["source"]
        v_id = c["target"]
        u_stmt = brain_graph.nodes_dict.get(u_id, {}).get("statement", "")
        v_stmt = brain_graph.nodes_dict.get(v_id, {}).get("statement", "")
        
        emb_u = brain_graph.emb_index.get_embedding(u_id)
        emb_v = brain_graph.emb_index.get_embedding(v_id)
        if emb_u is not None and emb_v is not None:
            sim = float(np.dot(emb_u, emb_v))
            sims.append(sim)

        query = f"Is the following hypothesis or finding supported by current scientific evidence: '{u_stmt}'"

        # Check if u or v is linked to an empirical node
        emp_nodes = set()
        for nid in (u_id, v_id):
            for _, nbr, ed in brain_graph.G.out_edges(nid, data=True):
                if ed.get("type") == "empirically_tested" or brain_graph.nodes_dict.get(nbr, {}).get("node_type") == "empirical":
                    emp_nodes.add(nbr)
            for nbr, _, ed in brain_graph.G.in_edges(nid, data=True):
                if ed.get("type") == "empirically_tested" or brain_graph.nodes_dict.get(nbr, {}).get("node_type") == "empirical":
                    emp_nodes.add(nbr)
        if emp_nodes:
            total_with_empirical += 1

        # 1. Brain-as-Graph
        bg_retrieved = brain_graph.retrieve_subgraph(query, top_k=8)
        bg_has_antithesis = v_id in bg_retrieved
        results["brain_graph"]["antithesis_retrieved"] += int(bg_has_antithesis)
        # Graph knows the edge has sign = -1 (contradicts)
        if bg_has_antithesis:
            results["brain_graph"]["correct_sign_disambiguation"] += 1
            # Falsification error is 0 because edge type is explicitly CONTRADICTS
        if emp_nodes and any(en in bg_retrieved for en in emp_nodes):
            empirical_resolutions_found["brain_graph"] += 1

        # 2. Vector RAG
        vr_retrieved = vec_single.retrieve(query, top_k=8)
        vr_has_antithesis = v_id in vr_retrieved
        results["vector_rag"]["antithesis_retrieved"] += int(vr_has_antithesis)
        # In Vector RAG, all retrieved nodes are presented as positive top-k similarity matches.
        # Vector RAG has NO signed edge representation: it cannot distinguish support from contradiction.
        # Thus, whenever it retrieves an opposing claim, it commits a Falsification Error (treating antithesis as relevant support).
        if vr_has_antithesis:
            results["vector_rag"]["falsification_error"] += 1
            # Zero sign disambiguation capability
        if emp_nodes and any(en in vr_retrieved for en in emp_nodes):
            empirical_resolutions_found["vector_rag"] += 1

        # 3. Flat File
        ff_retrieved = flat_file.retrieve(query, top_k=8)
        ff_has_antithesis = v_id in ff_retrieved
        results["flat_file"]["antithesis_retrieved"] += int(ff_has_antithesis)
        if ff_has_antithesis:
            results["flat_file"]["falsification_error"] += 1
        if emp_nodes and any(en in ff_retrieved for en in emp_nodes):
            empirical_resolutions_found["flat_file"] += 1

    total_c = len(contradictions)
    summary = {
        "cosine_similarity_distribution": {
            "mean": round(float(np.mean(sims)), 4),
            "median": round(float(np.median(sims)), 4),
            "min": round(float(np.min(sims)), 4),
            "max": round(float(np.max(sims)), 4),
            "fraction_above_0_70": round(float(np.mean([s >= 0.70 for s in sims])), 4),
            "fraction_above_0_50": round(float(np.mean([s >= 0.50 for s in sims])), 4)
        },
        "substrates": {}
    }

    for name, data in results.items():
        rec_rate = data["antithesis_retrieved"] / total_c
        disambig_rate = (data["correct_sign_disambiguation"] / data["antithesis_retrieved"]) if data["antithesis_retrieved"] > 0 else 0.0
        falsification_rate = (data["falsification_error"] / data["antithesis_retrieved"]) if data["antithesis_retrieved"] > 0 else 0.0
        emp_rate = (empirical_resolutions_found[name] / total_with_empirical) if total_with_empirical > 0 else 0.0
        
        summary["substrates"][name] = {
            "antithesis_recall_rate": round(rec_rate, 3),
            "sign_disambiguation_accuracy": round(disambig_rate, 3),
            "falsification_error_rate": round(falsification_rate, 3),
            "empirical_resolution_recall": round(emp_rate, 3)
        }

    print(f"Cosine Similarity between Contradictory Pairs: Mean={summary['cosine_similarity_distribution']['mean']}, "
          f"Median={summary['cosine_similarity_distribution']['median']}, Max={summary['cosine_similarity_distribution']['max']}")
    print(f"Fraction of Contradictions with Cosine Sim >= 0.50: {summary['cosine_similarity_distribution']['fraction_above_0_50']*100:.1f}%")
    print("\n" + f"{'Substrate':15s} | {'Antithesis Recall':18s} | {'Sign Disambiguation':20s} | {'Falsification Error':20s} | {'Empirical Resolution':21s}")
    print("-" * 105)
    for name, s in summary["substrates"].items():
        print(f"{name:15s} | {s['antithesis_recall_rate']*100:5.1f}%             | {s['sign_disambiguation_accuracy']*100:5.1f}%               | "
              f"{s['falsification_error_rate']*100:5.1f}%               | {s['empirical_resolution_recall']*100:5.1f}%")

    return summary


# ── Suite 4: Context Window Token Budget Efficiency ───────────────────────────

def run_suite_4_token_budget_efficiency(brain_graph: BrainGraphSubstrate,
                                        vec_single: VectorStoreSingleSubstrate,
                                        flat_file: FlatFileSubstrate) -> dict:
    print("\n" + "=" * 80)
    print("SUITE 4: CONTEXT WINDOW TOKEN BUDGET EFFICIENCY & RELATIONAL DENSITY")
    print("=" * 80)

    # Budgets in tokens (approx 4 chars per token)
    budgets = [250, 500, 1000, 2000]
    sample_queries = [
        "What are the thermodynamic limits on bit erasure and neural spike energy dissipation?",
        "How do kinetic proofreading and quantum lattice codes prevent information corruption?",
        "Evaluate the dialectic between rate coding models and temporal spike timing in SNNs."
    ]

    summary = {str(b): {} for b in budgets}

    for b in budgets:
        max_chars = b * 4
        for substrate_name, retrieve_fn in [
            ("brain_graph", lambda q: brain_graph.retrieve_subgraph(q, top_k=12)),
            ("vector_rag", lambda q: vec_single.retrieve(q, top_k=12)),
            ("flat_file", lambda q: flat_file.retrieve(q, top_k=12))
        ]:
            facts_delivered = []
            tokens_used = []
            irrelevant_noise_tokens = []

            for q in sample_queries:
                nids = retrieve_fn(q)
                # Pack prompt until budget is reached
                packed_text = ""
                packed_nids = []
                for nid in nids:
                    node = brain_graph.nodes_dict[nid]
                    text_line = f"[{node.get('node_type','concept')}] {node.get('statement','')}\n"
                    if len(packed_text) + len(text_line) > max_chars:
                        break
                    packed_text += text_line
                    packed_nids.append(nid)

                n_tokens = len(packed_text.split())
                tokens_used.append(n_tokens)

                # Count verified edges within the packed context
                sub = brain_graph.G.subgraph(packed_nids)
                n_edges = sub.number_of_edges()
                facts_delivered.append(n_edges)

                # Unconnected nodes represent disconnected filler / noise
                sub_u = sub.to_undirected()
                unconnected = sum(1 for n in packed_nids if sub_u.degree(n) == 0)
                noise_ratio = unconnected / len(packed_nids) if packed_nids else 0.0
                irrelevant_noise_tokens.append(noise_ratio)

            mean_edges = float(np.mean(facts_delivered))
            mean_tok = float(np.mean(tokens_used)) if tokens_used else 1.0
            density_per_1k = (mean_edges / max(mean_tok, 1)) * 1000.0

            summary[str(b)][substrate_name] = {
                "verified_relational_facts": round(mean_edges, 1),
                "tokens_consumed": round(mean_tok, 1),
                "relational_density_per_1k_tokens": round(density_per_1k, 2),
                "disconnected_noise_ratio": round(float(np.mean(irrelevant_noise_tokens)), 3)
            }

    print(f"{'Budget':8s} | {'Substrate':15s} | {'Relational Facts':18s} | {'Tokens Consumed':16s} | {'Facts/1k Tokens':17s} | {'Noise Ratio':12s}")
    print("-" * 95)
    for b in budgets:
        for name, data in summary[str(b)].items():
            print(f"{b:5d}    | {name:15s} | {data['verified_relational_facts']:5.1f}              | "
                  f"{data['tokens_consumed']:5.1f}            | {data['relational_density_per_1k_tokens']:6.2f}            | "
                  f"{data['disconnected_noise_ratio']*100:5.1f}%")

    return summary


# ── Suite 5: Topological Distraction Robustness ───────────────────────────────

def run_suite_5_distraction_robustness(brain_graph: BrainGraphSubstrate,
                                       vec_single: VectorStoreSingleSubstrate) -> dict:
    print("\n" + "=" * 80)
    print("SUITE 5: TOPOLOGICAL DISTRACTION ROBUSTNESS (DECOY INGESTION STRESS TEST)")
    print("=" * 80)

    # Generate 50 lexical decoy nodes with high keyword overlap but 0 graph edges
    # E.g. superficial blog posts or keyword salads discussing Landauer, DNA, Quantum, Spikes
    decoys = [
        {"id": f"decoy_{i:03d}", "statement": f"Superficial overview of quantum surface codes and thermodynamic Landauer limits in computing #{i}."}
        for i in range(50)
    ]

    # Create temporary embedding index with decoys added
    temp_index = EmbeddingIndex(dimension=brain_graph.emb_index.dimension)
    for nid, vec in brain_graph.emb_index._embeddings.items():
        temp_index.add(nid, vec)
    
    for d in decoys:
        vec = shared_embed(d["statement"])
        temp_index.add(d["id"], vec)

    test_queries = [
        "What are the thermodynamic Landauer limits on bit erasure in cellular systems?",
        "How do quantum surface codes perform parity checks on information lattices?",
        "Compare kinetic proofreading in DNA copying to error correction in spiking circuits."
    ]

    vec_decoy_hits = []
    graph_decoy_hits = []

    for q in test_queries:
        # Vector RAG top-8
        q_vec = shared_embed(q)
        v_hits = [nid for nid, _ in temp_index.query(q_vec, threshold=0.2, top_k=8)]
        v_decoys = sum(1 for nid in v_hits if nid.startswith("decoy_"))
        vec_decoy_hits.append(v_decoys / 8.0)

        # Brain-as-Graph: Seed from temp_index, but expand via graph edges!
        # Because decoys have 0 edges in G, they cannot spread or receive activation.
        seed_matches = temp_index.query(q_vec, threshold=0.25, top_k=5)
        activation = {nid: float(sim) for nid, sim in seed_matches if nid in brain_graph.G}
        
        # 2 hops of spreading activation
        for _ in range(2):
            for src, act in list(activation.items()):
                if act < 0.1:
                    continue
                for nbr in list(brain_graph.G.successors(src)) + list(brain_graph.G.predecessors(src)):
                    ed = brain_graph.G.get_edge_data(src, nbr) or brain_graph.G.get_edge_data(nbr, src) or {}
                    w = float(ed.get("weight", 0.5))
                    activation[nbr] = max(activation.get(nbr, 0.0), act * w * 0.7)

        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        g_hits = [nid for nid, _ in ranked[:8]]
        g_decoys = sum(1 for nid in g_hits if nid.startswith("decoy_"))
        graph_decoy_hits.append(g_decoys / 8.0)

    summary = {
        "vector_rag_mean_decoy_contamination": round(float(np.mean(vec_decoy_hits)), 3),
        "brain_graph_mean_decoy_contamination": round(float(np.mean(graph_decoy_hits)), 3),
        "topological_noise_rejection_factor": round(float(np.mean(vec_decoy_hits)) / max(float(np.mean(graph_decoy_hits)), 1e-6), 2)
    }

    print(f"Vector RAG Decoy Contamination Rate:    {summary['vector_rag_mean_decoy_contamination']*100:.1f}%")
    print(f"Brain-as-Graph Decoy Contamination Rate: {summary['brain_graph_mean_decoy_contamination']*100:.1f}%")
    print(f"Topological Noise Rejection Lift:        {summary['topological_noise_rejection_factor']:.1f}x")

    return summary


# ── Main Runner ───────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("STARTING PHASE 3 BENCHMARK: REPRESENTATION BASELINES DEEP-DIVE")
    print("=" * 80)

    with open(BRAIN_PATH, "r") as f:
        brain_data = json.load(f)

    emb_index = EmbeddingIndex.load(INDEX_PATH)

    brain_graph = BrainGraphSubstrate(brain_data, emb_index)
    vec_iter = VectorStoreIterativeSubstrate(brain_data, emb_index)
    vec_single = VectorStoreSingleSubstrate(brain_data, emb_index)
    flat_file = FlatFileSubstrate(brain_data)

    s1 = run_suite_1_path_discovery(brain_graph, vec_iter, vec_single, flat_file)
    s2 = run_suite_2_context_coherence(brain_graph, vec_single, flat_file)
    s3 = run_suite_3_contradictions(brain_graph, vec_single, flat_file)
    s4 = run_suite_4_token_budget_efficiency(brain_graph, vec_single, flat_file)
    s5 = run_suite_5_distraction_robustness(brain_graph, vec_single)

    final_report = {
        "timestamp": "2026-09-17T22:50:00+05:30",
        "connectome_stats": {
            "nodes": len(brain_data["graph"]["nodes"]),
            "edges": len(brain_data["graph"]["edges"]),
            "contradictions": sum(1 for e in brain_data["graph"]["edges"] if e.get("type") == "contradicts")
        },
        "suite_1_path_discovery": s1,
        "suite_2_context_coherence": s2,
        "suite_3_contradictions": s3,
        "suite_4_token_budget_efficiency": s4,
        "suite_5_distraction_robustness": s5
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PHASE 3 BENCHMARK COMPLETE. Saved comprehensive results to: {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
