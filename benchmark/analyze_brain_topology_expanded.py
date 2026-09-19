#!/usr/bin/env python3
"""
analyze_brain_topology_expanded.py

Expanded Topological and Connectome Horizon Analysis for AutoScientist.
Performs an exhaustive longitudinal network neuroscience analysis of the
knowledge graph across all 20 diurnal cycles.

Metrics Extracted:
1. Basic & Sparse Wiring: |V|, |E|, density, average degree, reciprocity
2. Connectome Efficiency: Global efficiency (Latora & Marchiori), Local efficiency
3. Small-World Architecture: Clustering C, Path length L, Small-World S = (C/C_rand)/(L/L_rand)
4. Functional Modularity: Louvain modularity Q, community count, community entropy, inter-community edge ratio
5. Scale-Free & Hub Emergence: Tail power-law exponent gamma, R^2, top hubs by PageRank, Betweenness, and Degree
6. Cognitive Node & Edge Dynamics: Node-type distribution, Edge-type distribution, Contradiction ratio, Empirical ratio
"""

import json
import math
import os
import sys
import time
import numpy as np
import networkx as nx

BRAIN_PATH = "data/brain.json"
CYCLE_LOG_PATH = "logs/cycle_log.json"
OUTPUT_DIR = "benchmark/results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "brain_topology_expanded.json")


def load_cycle_timestamps(cycle_log_path: str) -> dict:
    with open(cycle_log_path, 'r') as f:
        data = json.load(f)
    cycle_ends = {}
    for entry in data.get("entries", []):
        c = entry.get("cycle")
        t = entry.get("ended_at") or entry.get("started_at")
        if c and t:
            cycle_ends[c] = max(cycle_ends.get(c, 0.0), float(t))
    return cycle_ends


def reconstruct_subgraph_at_time(nodes: list, edges: list, timestamp: float) -> nx.DiGraph:
    G = nx.DiGraph()
    valid_nodes = set()
    for n in nodes:
        if n.get("created_at", 0) <= timestamp:
            nid = n["id"]
            valid_nodes.add(nid)
            G.add_node(nid, **{k: v for k, v in n.items() if k != "id"})

    for e in edges:
        if e.get("created_at", 0) <= timestamp:
            src = e.get("source")
            tgt = e.get("target")
            if src in valid_nodes and tgt in valid_nodes:
                G.add_edge(src, tgt, **{k: v for k, v in e.items() if k not in ("source", "target")})
    return G


def compute_efficiency_and_small_world(G_un: nx.Graph, n_random_samples: int = 5) -> dict:
    if len(G_un) < 4:
        return {
            "C_actual": 0.0, "L_actual": 0.0, "C_rand": 0.0, "L_rand": 0.0,
            "small_world_S": 0.0, "global_efficiency": 0.0, "local_efficiency": 0.0,
            "diameter": 0, "radius": 0
        }

    components = list(nx.connected_components(G_un))
    largest_cc = G_un.subgraph(max(components, key=len)).copy()
    n_nodes = largest_cc.number_of_nodes()
    n_edges = largest_cc.number_of_edges()

    # Clustering and Path Length
    C_actual = float(nx.average_clustering(largest_cc))
    try:
        L_actual = float(nx.average_shortest_path_length(largest_cc))
    except Exception:
        L_actual = 0.0

    # Global and Local Efficiency (Computational Neuroscience metrics)
    try:
        glob_eff = float(nx.global_efficiency(largest_cc))
    except Exception:
        glob_eff = 0.0

    try:
        loc_eff = float(nx.local_efficiency(largest_cc))
    except Exception:
        loc_eff = 0.0

    # Diameter and Radius on LCC
    try:
        diameter = int(nx.diameter(largest_cc))
        radius = int(nx.radius(largest_cc))
    except Exception:
        diameter = 0
        radius = 0

    # Erdos-Renyi Random Null Model (matched N and M)
    c_rands = []
    l_rands = []
    for _ in range(n_random_samples):
        R = nx.gnm_random_graph(n_nodes, n_edges)
        if nx.is_connected(R):
            c_rands.append(nx.average_clustering(R))
            l_rands.append(nx.average_shortest_path_length(R))
        else:
            r_comps = list(nx.connected_components(R))
            if r_comps:
                r_largest = R.subgraph(max(r_comps, key=len))
                if r_largest.number_of_nodes() > 3:
                    c_rands.append(nx.average_clustering(r_largest))
                    l_rands.append(nx.average_shortest_path_length(r_largest))

    C_rand = float(np.mean(c_rands)) if c_rands else (n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 1.0)
    L_rand = float(np.mean(l_rands)) if l_rands else (math.log(n_nodes) / math.log(max(2 * n_edges / n_nodes, 1.1)) if n_nodes > 1 else 1.0)

    # Small world index S
    if C_rand > 0 and L_actual > 0 and L_rand > 0:
        gamma = C_actual / C_rand
        lambda_val = L_actual / L_rand
        S = gamma / lambda_val if lambda_val > 0 else 0.0
    else:
        gamma = 1.0
        lambda_val = 1.0
        S = 0.0

    return {
        "C_actual": round(C_actual, 4),
        "L_actual": round(L_actual, 4),
        "C_rand": round(C_rand, 4),
        "L_rand": round(L_rand, 4),
        "gamma_clust": round(gamma, 4),
        "lambda_path": round(lambda_val, 4),
        "small_world_S": round(S, 4),
        "global_efficiency": round(glob_eff, 4),
        "local_efficiency": round(loc_eff, 4),
        "diameter": diameter,
        "radius": radius,
        "lcc_nodes": n_nodes,
        "lcc_edges": n_edges
    }


def compute_community_modularity(G_un: nx.Graph) -> dict:
    if len(G_un) < 4 or G_un.number_of_edges() < 2:
        return {"num_communities": 0, "modularity_Q": 0.0, "community_entropy": 0.0, "inter_community_edge_ratio": 0.0}

    try:
        communities = list(nx.community.louvain_communities(G_un, seed=42))
        Q = float(nx.community.modularity(G_un, communities))

        # Community sizes and Shannon entropy
        sizes = [len(c) for c in communities]
        total_nodes = sum(sizes)
        probs = [s / total_nodes for s in sizes]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Inter-community vs intra-community edges
        node_to_comm = {}
        for idx, comm in enumerate(communities):
            for n in comm:
                node_to_comm[n] = idx

        inter_edges = 0
        total_edges = G_un.number_of_edges()
        for u, v in G_un.edges():
            if node_to_comm.get(u) != node_to_comm.get(v):
                inter_edges += 1

        inter_ratio = inter_edges / total_edges if total_edges > 0 else 0.0

        return {
            "num_communities": len(communities),
            "modularity_Q": round(Q, 4),
            "community_entropy": round(entropy, 3),
            "inter_community_edge_ratio": round(inter_ratio, 4),
            "largest_community_size": max(sizes) if sizes else 0
        }
    except Exception as e:
        return {"num_communities": 0, "modularity_Q": 0.0, "error": str(e)}


def fit_scale_free_tail(degrees: list) -> dict:
    deg_counts = {}
    for d in degrees:
        if d > 0:
            deg_counts[d] = deg_counts.get(d, 0) + 1

    if len(deg_counts) < 4:
        return {"gamma": 0.0, "r_squared": 0.0}

    x = np.array(sorted(deg_counts.keys()))
    y = np.array([deg_counts[k] for k in x]) / float(len(degrees))

    log_x = np.log(x)
    log_y = np.log(y)

    coeffs = np.polyfit(log_x, log_y, 1)
    slope = coeffs[0]
    gamma = -slope

    p = np.poly1d(coeffs)
    yhat = p(log_x)
    ybar = np.mean(log_y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((log_y - ybar) ** 2)
    r_squared = ssreg / sstot if sstot > 0 else 0.0

    return {
        "gamma": round(float(gamma), 3),
        "r_squared": round(float(r_squared), 3)
    }


def find_expanded_hubs(G_di: nx.DiGraph, top_n: int = 5) -> list:
    if len(G_di) == 0:
        return []

    degrees = dict(G_di.degree())
    try:
        pageranks = nx.pagerank(G_di, max_iter=200)
    except Exception:
        pageranks = {n: 0.0 for n in G_di}

    try:
        betweenness = nx.betweenness_centrality(G_di, k=min(100, len(G_di)))
    except Exception:
        betweenness = {n: 0.0 for n in G_di}

    # Combined hub ranking
    sorted_nodes = sorted(degrees.items(), key=lambda x: (pageranks.get(x[0], 0), x[1]), reverse=True)[:top_n]
    hubs = []
    for nid, deg in sorted_nodes:
        node_data = G_di.nodes.get(nid, {})
        stmt = node_data.get("statement", "") or node_data.get("name", "")
        ntype = node_data.get("node_type", "")
        hubs.append({
            "id": nid[:8],
            "node_type": ntype,
            "degree": deg,
            "pagerank": round(float(pageranks.get(nid, 0.0)), 5),
            "betweenness": round(float(betweenness.get(nid, 0.0)), 5),
            "statement": stmt[:85] + "..." if len(stmt) > 85 else stmt
        })
    return hubs


def run_expanded_analysis():
    print("=" * 80)
    print("AUTOSCIENTIST: EXPANDED CONNECTOME & GRAPH HORIZON ANALYSIS")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(BRAIN_PATH, 'r') as f:
        brain_data = json.load(f)

    nodes = brain_data["graph"].get("nodes", [])
    edges = brain_data["graph"].get("edges", [])
    print(f"Master Graph Loaded: {len(nodes)} nodes, {len(edges)} edges.")

    cycle_ends = load_cycle_timestamps(CYCLE_LOG_PATH)
    sorted_cycles = sorted(cycle_ends.keys())
    print(f"Longitudinal Span: {len(sorted_cycles)} completed diurnal cycles.\n")

    results = []

    for c in sorted_cycles:
        t_end = cycle_ends[c]
        G_di = reconstruct_subgraph_at_time(nodes, edges, t_end)
        G_un = G_di.to_undirected()

        n_nodes = G_di.number_of_nodes()
        n_edges = G_di.number_of_edges()

        if n_nodes == 0:
            continue

        density = round(float(nx.density(G_di)), 5)
        reciprocity = round(float(nx.reciprocity(G_di)), 4) if n_edges > 0 else 0.0
        degrees = [d for _, d in G_di.degree()]
        avg_degree = round(float(np.mean(degrees)), 2)

        # Node type proportions
        node_types = {}
        for _, d in G_di.nodes(data=True):
            nt = d.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        # Edge type proportions
        edge_types = {}
        for _, _, d in G_di.edges(data=True):
            et = d.get("type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        # Cognitive metrics
        n_hyp = node_types.get("hypothesis", 0)
        n_emp = node_types.get("empirical", 0)
        n_contra = edge_types.get("contradicts", 0)
        emp_ratio = round(n_emp / n_hyp, 3) if n_hyp > 0 else 0.0
        contra_ratio = round(n_contra / n_edges, 4) if n_edges > 0 else 0.0

        # Topological efficiency and small-world
        eff_sw = compute_efficiency_and_small_world(G_un, n_random_samples=5)

        # Modularity & communities
        comm_stats = compute_community_modularity(G_un)

        # Scale-free power law
        tail_fit = fit_scale_free_tail(degrees)

        # Top cortical hubs
        top_hubs = find_expanded_hubs(G_di, top_n=5)

        snapshot = {
            "cycle": c,
            "timestamp": t_end,
            "graph_size": {
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "avg_degree": avg_degree,
                "reciprocity": reciprocity
            },
            "cognitive_ratios": {
                "empirical_ratio": emp_ratio,
                "contradiction_ratio": contra_ratio,
                "node_types": node_types,
                "edge_types": edge_types
            },
            "connectome_topology": eff_sw,
            "community_structure": comm_stats,
            "scale_free_fit": tail_fit,
            "cortical_hubs": top_hubs
        }
        results.append(snapshot)

        print(f"Cycle {c:2d} | Nodes: {n_nodes:3d} | Edges: {n_edges:4d} | "
              f"Density: {density:.4f} | Modularity Q: {comm_stats.get('modularity_Q', 0.0):.3f} | "
              f"GlobEff: {eff_sw.get('global_efficiency', 0.0):.3f} | Path L: {eff_sw.get('L_actual', 0.0):.2f} | "
              f"Small-World S: {eff_sw.get('small_world_S', 0.0):5.2f} | Empirical: {n_emp:2d} | Contradictions: {n_contra:2d}")

    # Write full structured record
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({"trajectory": results}, f, indent=2)

    print(f"\nSaved expanded connectome trajectory to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_expanded_analysis()
