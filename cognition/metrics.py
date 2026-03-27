from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def activation_entropy(activations: Dict[str, float]) -> float:
    vals = [max(0.0, v) for v in activations.values()]
    total = sum(vals)
    if total <= 1e-12:
        return 0.0
    ps = [v / total for v in vals if v > 0]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def contradiction_density(total_edges: int, contradiction_edges: int) -> float:
    return contradiction_edges / max(1, total_edges)


def cluster_modularity_like(clusters: Dict[str, List[str]], edges: Iterable[Tuple[str, str]]) -> float:
    edge_list = list(edges)
    total = len(edge_list)
    if total == 0:
        return 0.0
    membership = {}
    for c, ids in clusters.items():
        for nid in ids:
            membership[nid] = c
    internal = sum(1 for u, v in edge_list if membership.get(u) == membership.get(v) and membership.get(u) is not None)
    return internal / total


def compression_gain(before_nodes: int, before_edges: int, after_nodes: int, after_edges: int) -> float:
    before = before_nodes + before_edges
    after = after_nodes + after_edges
    return (before - after) / max(1, before)


def uncertainty_mass(node_uncertainties: Dict[str, float]) -> float:
    return float(sum(node_uncertainties.values()))


def test_informativeness(outcomes: List[Dict]) -> float:
    if not outcomes:
        return 0.0
    return sum(float(o.get("uncertainty_reduction", 0.0)) for o in outcomes) / len(outcomes)


def event_rates(events: List[Dict], window: int = 500) -> Dict[str, float]:
    tail = events[-window:]
    counts = defaultdict(int)
    for e in tail:
        counts[e.get("type", "unknown")] += 1
    denom = max(1, len(tail))
    return {k: v / denom for k, v in counts.items()}
