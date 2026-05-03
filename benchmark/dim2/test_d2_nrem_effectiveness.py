"""
Dimension 2 — Test 5: NREM Effectiveness
========================================
Measures whether the NREM pass reinforces edges aligned with the replayed
episodic trajectory, rather than merely selecting globally important edges.

This benchmark matches the actual runtime:
- `dreamer.nrem_pass()` replays a recent episodic sequence
- replay updates node activation
- `brain.proximal_reinforce()` then boosts prioritized strong edges

Pass criterion:
- reinforced candidate edges must show higher replay-node overlap than
  eligible unreinforced controls
- reinforced candidate edges must contain a higher fraction of replay-path
  edges than eligible unreinforced controls
- the benchmark must actually exercise replay-aligned candidate edges

Benchmark level:
  - module-level

Usage:
    python benchmark/dim2/test_d2_nrem_effectiveness.py \
        --out benchmark/dim2/results/d2_nrem_effectiveness.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORPUS = [
    {"id": "dna", "title": "DNA"},
    {"id": "thermodynamics", "title": "Thermodynamics"},
    {"id": "natural_selection", "title": "Natural selection"},
    {"id": "neural_network", "title": "Artificial neural network"},
    {"id": "game_theory", "title": "Game theory"},
]


def fetch_wikipedia(title: str) -> str:
    import requests

    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "format": "json",
        "redirects": 1,
    }
    resp = requests.get(
        api,
        params=params,
        timeout=20,
        headers={"User-Agent": "AutoScientist-Benchmark/2.0"},
    )
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")[:8000]
    return ""


def _edge_key(u, v):
    return tuple(sorted((u, v)))


def _extract_replay_nodes(log, max_nodes: int = 4):
    replay_nodes = []
    for step in log.steps:
        for nid in [step.from_id, step.to_id]:
            if nid and nid not in replay_nodes:
                replay_nodes.append(nid)
            if len(replay_nodes) >= max_nodes:
                return replay_nodes
    return replay_nodes


def _fallback_replay_nodes(brain, max_nodes: int = 4):
    ranked = sorted(brain.graph.degree, key=lambda item: item[1], reverse=True)
    return [nid for nid, _ in ranked[:max_nodes]]


def _collect_candidate_edges(brain, weights_before, replay_nodes, replay_path_edges, threshold):
    replay_set = set(replay_nodes)
    candidates = []
    for u, v, data in brain.graph.edges(data=True):
        weight_before = weights_before.get((u, v), data.get("weight", 0.5))
        if weight_before < threshold:
            continue
        node_u = brain.get_node(u) or {}
        node_v = brain.get_node(v) or {}
        replay_overlap = int(u in replay_set) + int(v in replay_set)
        candidates.append({
            "edge_u": u,
            "edge_v": v,
            "statement_u": node_u.get("statement", "")[:140],
            "statement_v": node_v.get("statement", "")[:140],
            "edge_type": data.get("type", "associated"),
            "weight_before": round(weight_before, 3),
            "replay_overlap": replay_overlap,
            "touches_replay": replay_overlap > 0,
            "is_replay_path_edge": _edge_key(u, v) in replay_path_edges,
        })
    return candidates


def _mean(items, field):
    if not items:
        return 0.0
    return sum(item[field] for item in items) / len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--out", default="benchmark/dim2/results/d2_nrem_effectiveness.json")
    parser.add_argument("--candidate-threshold", type=float, default=0.60)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from graph.brain import Brain, EdgeSource
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer
    from dreamer.dreamer import Dreamer

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)

    print("=" * 60)
    print("PHASE 1: Loading or building shared graph")
    print("=" * 60)

    brain_path = "benchmark/dim2/shared/brain.json"
    index_path = "benchmark/dim2/shared/embedding_index"
    if os.path.exists(brain_path) and os.path.exists(index_path + ".json"):
        print("  Loading shared brain and index...")
        brain.load(brain_path)
        emb_index = EmbeddingIndex.load(index_path)
    else:
        print("  Shared graph not found. Ingesting benchmark corpus...")
        observer = Observer(brain)
        ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)
        for article in CORPUS:
            print(f"  Ingesting: {article['title']}...")
            text = fetch_wikipedia(article["title"])
            if text:
                ingestor.ingest(text, source=EdgeSource.READING)
                time.sleep(1)

    observer = Observer(brain)
    dreamer = Dreamer(brain, research_agenda=observer)

    print("\n" + "=" * 60)
    print("PHASE 2: Running REM cycles to create a recent trajectory")
    print("=" * 60)
    last_log = None
    for i in range(3):
        print(f"  Dream cycle {i + 1}/3...")
        last_log = dreamer.dream(steps=20, run_nrem=False)

    replay_nodes = _extract_replay_nodes(last_log or dreamer.dream(steps=8, run_nrem=False))
    if len(replay_nodes) < 3:
        replay_nodes = _fallback_replay_nodes(brain)

    replay_path_edges = {
        _edge_key(replay_nodes[i], replay_nodes[i + 1])
        for i in range(len(replay_nodes) - 1)
    }

    print("\n" + "=" * 60)
    print("PHASE 3: Seeding replay event and collecting candidate edges")
    print("=" * 60)
    print(f"  Replay nodes: {len(replay_nodes)}")
    brain.episodic.record(
        "benchmark_replay",
        "Recent dream trajectory for NREM replay benchmark",
        replay_nodes,
    )

    weights_before = {
        (u, v): data.get("weight", 0.5)
        for u, v, data in brain.graph.edges(data=True)
    }
    candidates = _collect_candidate_edges(
        brain,
        weights_before,
        replay_nodes,
        replay_path_edges,
        args.candidate_threshold,
    )
    candidate_replay_path_edges = sum(1 for item in candidates if item["is_replay_path_edge"])
    print(f"  Candidate edges: {len(candidates)}")
    print(f"  Candidate replay-path edges: {candidate_replay_path_edges}")

    print("\n" + "=" * 60)
    print("PHASE 4: Running NREM pass")
    print("=" * 60)
    dreamer.nrem_pass()

    reinforced = []
    controls = []
    for item in candidates:
        edge = brain.get_edge(item["edge_u"], item["edge_v"]) or {}
        weight_after = edge.get("weight", item["weight_before"])
        enriched = dict(item)
        enriched["weight_after"] = round(weight_after, 3)
        enriched["reinforced"] = weight_after > item["weight_before"]
        if enriched["reinforced"]:
            reinforced.append(enriched)
        else:
            controls.append(enriched)

    mean_replay_overlap_reinforced = _mean(reinforced, "replay_overlap")
    mean_replay_overlap_controls = _mean(controls, "replay_overlap")
    replay_touch_fraction_reinforced = _mean(reinforced, "touches_replay")
    replay_touch_fraction_controls = _mean(controls, "touches_replay")
    replay_path_fraction_reinforced = _mean(reinforced, "is_replay_path_edge")
    replay_path_fraction_controls = _mean(controls, "is_replay_path_edge")

    benchmark_exercised = (
        len(reinforced) > 0 and
        len(controls) > 0 and
        candidate_replay_path_edges > 0
    )

    passed = (
        benchmark_exercised and
        mean_replay_overlap_reinforced > mean_replay_overlap_controls and
        replay_touch_fraction_reinforced > replay_touch_fraction_controls and
        replay_path_fraction_reinforced > replay_path_fraction_controls
    )

    report = {
        "test": "D2 — NREM Effectiveness",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
            "candidate_threshold": args.candidate_threshold,
        },
        "summary": {
            "candidate_edges": len(candidates),
            "candidate_replay_path_edges": candidate_replay_path_edges,
            "reinforced_edges_evaluated": len(reinforced),
            "unreinforced_edges_evaluated": len(controls),
            "mean_replay_overlap_reinforced": round(mean_replay_overlap_reinforced, 3),
            "mean_replay_overlap_unreinforced": round(mean_replay_overlap_controls, 3),
            "replay_touch_fraction_reinforced": round(replay_touch_fraction_reinforced, 3),
            "replay_touch_fraction_unreinforced": round(replay_touch_fraction_controls, 3),
            "replay_path_fraction_reinforced": round(replay_path_fraction_reinforced, 3),
            "replay_path_fraction_unreinforced": round(replay_path_fraction_controls, 3),
            "benchmark_exercised": benchmark_exercised,
            "PASS": passed,
        },
        "replay_nodes": replay_nodes,
        "replay_path_edges": sorted(replay_path_edges),
        "reinforced_edges": reinforced,
        "unreinforced_edges": controls,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D2: NREM Effectiveness")
    print("=" * 60)
    print(f"Candidate edges                : {len(candidates)}")
    print(f"Reinforced candidate edges     : {len(reinforced)}")
    print(f"Unreinforced candidate edges   : {len(controls)}")
    print(f"Replay overlap (reinforced)    : {mean_replay_overlap_reinforced:.3f}")
    print(f"Replay overlap (controls)      : {mean_replay_overlap_controls:.3f}")
    print(f"Replay-path frac (reinforced)  : {replay_path_fraction_reinforced:.2%}")
    print(f"Replay-path frac (controls)    : {replay_path_fraction_controls:.2%}")
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
