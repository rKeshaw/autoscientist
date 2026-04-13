"""
Dimension 5 - Test 4: Reading List Quality
============================================
Tests whether the Reader generates sensible reading suggestions given the
current graph state.

Tests both mission-directed and wandering modes.

Benchmark level:
  - module-level
"""

import json
import os
import sys
import time
import tempfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import get_isolated_graph, judge_json, make_reader


# ── Judge prompts ────────────────────────────────────────────────────────────

READING_JUDGE = """
You are evaluating the quality of a reading suggestion for a scientific
knowledge graph.

MODE: {mode}
{mission_line}

CURRENT KNOWLEDGE DOMAINS: {domains}

SUGGESTED READING:
  Title: "{title}"
  URL: "{url}"
  Reason: "{reason}"

Evaluate:
1. **Relevance**: How relevant is this to the {mode_context}?
2. **Novelty**: Does this cover ground NOT already in the knowledge graph?
3. **Value**: Would reading this meaningfully expand the graph's knowledge?

Score each 1-7:
- relevance: 1 to 7
- novelty: 1 to 7
- value: 1 to 7
- is_redundant: true if the graph already covers this topic well

Respond EXACTLY in JSON:
{{
  "relevance": 1 to 7,
  "novelty": 1 to 7,
  "value": 1 to 7,
  "is_redundant": true or false,
  "reasoning": "one sentence"
}}
"""


def _extract_domains(brain):
    """Get the set of knowledge domains from the graph."""
    clusters = set()
    for _, data in brain.all_nodes():
        c = data.get("cluster", "")
        if c and c != "unclustered":
            clusters.add(c)
    return sorted(clusters)[:15]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim5/results/d5_reading_list.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 4: Reading List Quality")
    print("=" * 60)

    MISSION = (
        "How do biological and artificial learning systems balance "
        "exploration with stability?"
    )

    evaluations = []
    total_score = 0.0
    total_count = 0
    redundant_count = 0

    for mode in ["mission", "wandering"]:
        print(f"\n  Mode: {mode}")

        import reader.reader as reader_module

        brain, emb_index = get_isolated_graph(MISSION if mode == "mission" else None)
        domains = _extract_domains(brain)

        if mode == "wandering":
            from graph.brain import BrainMode
            brain.set_mode(BrainMode.WANDERING)

        original_reading_list_path = reader_module.READING_LIST_PATH
        with tempfile.TemporaryDirectory(prefix="d5_reading_list_") as tmpdir:
            reader_module.READING_LIST_PATH = os.path.join(tmpdir, "reading_list.json")
            try:
                reader, brain, emb_index, _ = make_reader(brain=brain, emb_index=emb_index)
                suggestions = reader.generate_reading_list()
            finally:
                reader_module.READING_LIST_PATH = original_reading_list_path

        print(f"    Generated {len(suggestions)} suggestions")

        mission = brain.get_mission()
        mission_text = mission["question"] if mission else "None"

        mode_evals = []
        for s in suggestions:
            title = getattr(s, "title", "")
            url = getattr(s, "url", "")
            reason = getattr(s, "added_reason", "")

            if not title:
                continue

            total_count += 1

            judgment = judge_json(
                READING_JUDGE.format(
                    mode=mode,
                    mission_line=f'MISSION: "{mission_text}"' if mode == "mission" else "",
                    domains=", ".join(domains),
                    title=title,
                    url=url,
                    reason=reason,
                    mode_context="research mission" if mode == "mission" else "intellectual curiosity",
                ),
                model=args.judge_model,
                default={
                    "relevance": 1, "novelty": 1, "value": 1,
                    "is_redundant": False, "reasoning": "Judge parse failed",
                },
            )

            relevance = float(judgment.get("relevance", 1))
            novelty = float(judgment.get("novelty", 1))
            value = float(judgment.get("value", 1))
            score = (relevance + novelty + value) / 3.0
            total_score += score
            is_redundant = bool(judgment.get("is_redundant", False))
            if is_redundant:
                redundant_count += 1

            mode_evals.append({
                "title": title,
                "url": url,
                "reason": reason,
                "relevance": relevance,
                "novelty": novelty,
                "value": value,
                "score": round(score, 3),
                "is_redundant": is_redundant,
                "judgment": judgment,
            })

            status = "✓" if score >= 4.5 else "~" if score >= 3.0 else "✗"
            redundant_tag = " [REDUNDANT]" if is_redundant else ""
            print(f"    {status} [{score:.1f}/7] {title[:50]}{redundant_tag}")

        evaluations.append({
            "mode": mode,
            "suggestions_count": len(mode_evals),
            "mean_score": round(sum(e["score"] for e in mode_evals) / max(len(mode_evals), 1), 3),
            "redundant_count": sum(1 for e in mode_evals if e["is_redundant"]),
            "suggestions": mode_evals,
        })
        time.sleep(0.2)

    mean_score = total_score / max(total_count, 1)
    passed = (
        total_count > 0 and
        mean_score >= 4.5 and
        redundant_count <= 2
    )

    report = {
        "test": "D5 - Reading List Quality",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model},
        "summary": {
            "total_suggestions": total_count,
            "mean_score": round(mean_score, 3),
            "redundant_count": redundant_count,
            "benchmark_exercised": total_count > 0,
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nMean score      : {mean_score:.2f}/7")
    print(f"Redundant count : {redundant_count}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
