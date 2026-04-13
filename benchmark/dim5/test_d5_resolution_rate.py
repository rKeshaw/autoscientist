"""
Dimension 5 - Test 3: Resolution Rate
========================================
Tests whether the Researcher's end-to-end pipeline actually advances
research questions — from query generation through web/arXiv search
to ingestion and resolution checking.

Requires network access (DuckDuckGo, arXiv).

Benchmark level:
  - pipeline-level
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

from _shared import judge_json, make_researcher


RESOLUTION_JUDGE = """
You are evaluating whether research findings meaningfully advanced a question.

RESEARCH QUESTION: "{question}"

RESEARCHER'S RESOLUTION GRADE: "{resolution_grade}"
QUERIES USED:
{queries}

SOURCE URLS CONSULTED:
{sources}

EVIDENCE NODES ADDED:
{evidence_nodes}

Was the question meaningfully advanced by this research?
Consider:
1. Do the evidence nodes contain specific information relevant to the question?
2. Was the resolution grade reasonable given the evidence shown?
3. Would a human researcher consider this a productive research session?
4. If there are no meaningful evidence nodes, mark advanced=false.

Score:
- advancement: 1 to 7 (was the question actually advanced?)
- efficiency: 1 to 7 (was the research process efficient?)

Respond EXACTLY in JSON:
{{
  "advanced": true or false,
  "advancement": 1 to 7,
  "efficiency": 1 to 7,
  "reasoning": "one sentence"
}}
"""


# Questions that should be answerable via web search
QUESTIONS = [
    (
        "What is the relationship between Hebbian learning and synaptic plasticity?",
        "Well-studied topic — should get strong resolution",
    ),
    (
        "How do genetic algorithms use crossover and mutation operators?",
        "Standard CS topic — should get partial or strong resolution",
    ),
    (
        "What is the role of dopamine in reinforcement learning in the brain?",
        "Neuroscience topic with good web coverage",
    ),
    (
        "How does the no-free-lunch theorem limit optimization algorithms?",
        "Mathematical CS topic — should be answerable",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--skip-network", action="store_true", default=False)
    parser.add_argument(
        "--out",
        default="benchmark/dim5/results/d5_resolution_rate.json",
    )
    args = parser.parse_args()

    if args.skip_network:
        print("TEST 3: Resolution Rate — SKIPPED (--skip-network)")
        report = {
            "test": "D5 - Resolution Rate",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {
                "PASS": False,
                "skipped": True,
                "skip_reason": "Network-dependent benchmark skipped by user request.",
            },
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("TEST 3: Resolution Rate (live web + arXiv search)")
    print("=" * 60)

    evaluations = []
    resolved_count = 0
    judged_advanced_count = 0
    total_advancement = 0.0

    for question_text, description in QUESTIONS:
        print(f"\n  Question: {question_text[:60]}...")
        print(f"  ({description})")

        # Each question gets its own isolated graph copy
        researcher, brain, emb_index, observer = make_researcher(depth="standard")

        # Add question to the observer's agenda
        observer.add_to_agenda(text=question_text)

        # Research the question
        entry = researcher._research_question(question_text)

        print(f"    Queries used: {len(entry.queries)}")
        print(f"    Sources found: {len(entry.sources)}")
        print(f"    Nodes created: {len(entry.node_ids)}")
        print(f"    Resolution: {entry.resolved}")

        is_resolved = entry.resolved in ("partial", "strong")
        if is_resolved:
            resolved_count += 1

        evidence_nodes = []
        for node_id in entry.node_ids:
            node = brain.get_node(node_id)
            if node:
                evidence_nodes.append(node.get("statement", "")[:300])

        evidence_block = (
            "\n".join(f"- {stmt}" for stmt in evidence_nodes[:8])
            if evidence_nodes else
            "- none"
        )
        source_block = (
            "\n".join(f"- {src}" for src in entry.sources[:8])
            if entry.sources else
            "- none"
        )
        query_block = (
            "\n".join(f"- {q}" for q in entry.queries)
            if entry.queries else
            "- none"
        )

        judgment = judge_json(
            RESOLUTION_JUDGE.format(
                question=question_text,
                resolution_grade=entry.resolved,
                queries=query_block,
                sources=source_block,
                evidence_nodes=evidence_block,
            ),
            model=args.judge_model,
            default={
                "advanced": False, "advancement": 1,
                "efficiency": 1, "reasoning": "Judge parse failed",
            },
        )

        judged_advanced = bool(judgment.get("advanced", False))
        if judged_advanced:
            judged_advanced_count += 1
        advancement = float(judgment.get("advancement", 1))
        total_advancement += advancement

        evaluations.append({
            "question": question_text,
            "description": description,
            "queries": entry.queries,
            "source_urls": entry.sources,
            "evidence_nodes": evidence_nodes,
            "sources_count": len(entry.sources),
            "nodes_created": len(entry.node_ids),
            "resolution_grade": entry.resolved,
            "is_resolved": is_resolved,
            "judged_advanced": judged_advanced,
            "judgment": judgment,
        })

        status = "✓" if judged_advanced else "✗"
        print(f"    {status} Resolution: {entry.resolved}, "
              f"Advancement: {advancement:.0f}/7")

        time.sleep(2)  # rate limiting

    n = max(len(QUESTIONS), 1)
    resolution_rate = resolved_count / n
    judged_advanced_rate = judged_advanced_count / n
    mean_advancement = total_advancement / n

    passed = (
        resolution_rate >= 0.60 and
        judged_advanced_rate >= 0.60 and
        mean_advancement >= 4.0
    )

    report = {
        "test": "D5 - Resolution Rate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model, "skip_network": False},
        "summary": {
            "questions_evaluated": len(QUESTIONS),
            "resolved_count": resolved_count,
            "resolution_rate": round(resolution_rate, 3),
            "judged_advanced_count": judged_advanced_count,
            "judged_advanced_rate": round(judged_advanced_rate, 3),
            "mean_advancement": round(mean_advancement, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nResolution rate     : {resolution_rate:.2%}")
    print(f"Judged advanced rate: {judged_advanced_rate:.2%}")
    print(f"Mean advancement    : {mean_advancement:.2f}/7")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
