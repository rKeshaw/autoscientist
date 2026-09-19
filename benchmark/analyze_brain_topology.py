#!/usr/bin/env python3
"""
analyze_brain_topology.py

Rigorous topological and connectome analysis of AutoScientist's knowledge graph
across its full longitudinal execution history (Cycles 1 to 20).

Calculates:
- Graph growth: |V|, |E|, average degree, density
- Small-worldness index S = (C / C_rand) / (L / L_rand) via degree-matched null models
- Modularity Q and community structure via Louvain detection
- Degree distribution and scale-free power-law exponent gamma
- Hub identification via degree and betweenness centrality
- Distribution of cognitive node and edge types over time
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
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "brain_topology_trajectory.json")


def load_cycle_timestamps(cycle_log_path: str) -> dict:
    """Extract the timestamp marking the end of each cycle."""
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
    """Build the directed NetworkX graph as it existed at `timestamp`."""
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


def compute_small_worldness(G_undirected: nx.Graph, n_random_samples: int = 5) -> dict:
    """
    Compute clustering coefficient C, path length L, and small-world coefficient S.
    Uses the largest connected component of the graph.
    Null model: G(n, m) with identical node and edge counts.
    """
    if len(G_undirected) < 4:
        return {"C": 0.0, "L": 0.0, "C_rand": 0.0, "L_rand": 0.0, "S": 0.0}

    # Extract largest connected component
    components = list(nx.connected_components(G_undirected))
    largest_cc = G_undirected.subgraph(max(components, key=len)).copy()
    n_nodes = largest_cc.number_of_nodes()
    n_edges = largest_cc.number_of_edges()

    if n_nodes < 4 or n_edges < n_nodes - 1:
        return {"C": 0.0, "L": 0.0, "C_rand": 0.0, "L_rand": 0.0, "S": 0.0}

    # Actual clustering and average shortest path
    C_actual = float(nx.average_clustering(largest_cc))
    try:
        L_actual = float(nx.average_shortest_path_length(largest_cc))
    except Exception:
        L_actual = 0.0

    # Random null models (Erdos-Renyi with matched n and m)
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

    # Small world index S = (C / C_rand) / (L / L_rand)
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
        "lcc_nodes": n_nodes,
        "lcc_edges": n_edges
    }


def fit_power_law(degrees: list) -> dict:
    """Fit degree distribution tail to P(k) ~ k^(-gamma) via log-log linear regression."""
    deg_counts = {}
    for d in degrees:
        if d > 0:
            deg_counts[d] = deg_counts.get(d, 0) + 1

    if len(deg_counts) < 4:
        return {"gamma": 0.0, "r_squared": 0.0}

    x = np.array(sorted(deg_counts.keys()))
    y = np.array([deg_counts[k] for k in x]) / float(len(degrees))

    # Log-log linear regression
    log_x = np.log(x)
    log_y = np.log(y)

    coeffs = np.polyfit(log_x, log_y, 1)
    slope = coeffs[0]
    gamma = -slope

    # R^2 calculation
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


def compute_modularity(G_undirected: nx.Graph) -> dict:
    """Compute community modularity Q using Louvain algorithm."""
    if len(G_undirected) < 4 or G_undirected.number_of_edges() < 2:
        return {"num_communities": 0, "modularity_Q": 0.0}

    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
        Q = nx.community.modularity(G_undirected, communities)
        return {
            "num_communities": len(communities),
            "modularity_Q": round(float(Q), 4)
        }
    except Exception as e:
        return {"num_communities": 0, "modularity_Q": 0.0, "error": str(e)}


def find_top_hubs(G: nx.DiGraph, top_n: int = 5) -> list:
    """Identify cortical hub nodes by total degree and degree centrality."""
    hubs = []
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for nid, deg in sorted_nodes:
        node_data = G.nodes.get(nid, {})
        stmt = node_data.get("statement", "")
        ntype = node_data.get("node_type", "")
        hubs.append({
            "id": nid[:8],
            "degree": deg,
            "node_type": ntype,
            "statement": stmt[:80] + "..." if len(stmt) > 80 else stmt
        })
    return hubs


def analyze_all_cycles():
    print("=" * 70)
    print("AUTOSCIENTIST: CONNECTOME & KNOWLEDGE GRAPH TOPOLOGY ANALYSIS")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(BRAIN_PATH):
        print(f"Error: {BRAIN_PATH} not found.")
        sys.exit(1)
    if not os.path.exists(CYCLE_LOG_PATH):
        print(f"Error: {CYCLE_LOG_PATH} not found.")
        sys.exit(1)

    print("Loading full brain graph and cycle logs...")
    with open(BRAIN_PATH, 'r') as f:
        brain_data = json.load(f)

    nodes = brain_data["graph"].get("nodes", [])
    edges = brain_data["graph"].get("edges", [])
    print(f"Loaded master graph: {len(nodes)} nodes, {len(edges)} edges.")

    cycle_ends = load_cycle_timestamps(CYCLE_LOG_PATH)
    sorted_cycles = sorted(cycle_ends.keys())
    print(f"Found {len(sorted_cycles)} completed cycles (Cycles {sorted_cycles[0]} to {sorted_cycles[-1]}).\n")

    results = []

    for c in sorted_cycles:
        t_end = cycle_ends[c]
        G_di = reconstruct_subgraph_at_time(nodes, edges, t_end)
        G_un = G_di.to_undirected()

        n_nodes = G_di.number_of_nodes()
        n_edges = G_di.number_of_edges()

        if n_nodes == 0:
            continue

        # Density and degree
        density = round(float(nx.density(G_di)), 5)
        degrees = [d for _, d in G_di.degree()]
        avg_degree = round(float(np.mean(degrees)), 2)

        # Node type distribution
        node_types = {}
        for _, d in G_di.nodes(data=True):
            nt = d.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        # Edge type distribution
        edge_types = {}
        for _, _, d in G_di.edges(data=True):
            et = d.get("type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        # Small-world metrics
        sw_metrics = compute_small_worldness(G_un, n_random_samples=5)

        # Modularity
        mod_metrics = compute_modularity(G_un)

        # Power law degree fit
        tail_fit = fit_power_law(degrees)

        # Top hubs
        top_hubs = find_top_hubs(G_di, top_n=3)

        cycle_stat = {
            "cycle": c,
            "timestamp": t_end,
            "nodes": n_nodes,
            "edges": n_edges,
            "density": density,
            "avg_degree": avg_degree,
            "node_types": node_types,
            "edge_types": edge_types,
            "small_world": sw_metrics,
            "modularity": mod_metrics,
            "power_law": tail_fit,
            "top_hubs": top_hubs
        }
        results.append(cycle_stat)

        print(f"Cycle {c:2d} | Nodes: {n_nodes:3d} | Edges: {n_edges:4d} | "
              f"Density: {density:.4f} | Modularity Q: {mod_metrics.get('modularity_Q', 0.0):.3f} | "
              f"Clust C: {sw_metrics.get('C_actual', 0.0):.3f} | Path L: {sw_metrics.get('L_actual', 0.0):.2f} | "
              f"Small-World S: {sw_metrics.get('small_world_S', 0.0):.2f} | gamma: {tail_fit.get('gamma', 0.0):.2f}")

    # Save structured results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({"trajectory": results}, f, indent=2)
    print(f"\nSaved full longitudinal topology trajectory to: {OUTPUT_JSON}")


if __name__ == "__main__":
    analyze_all_cycles()
