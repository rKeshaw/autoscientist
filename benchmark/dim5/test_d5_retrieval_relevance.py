"""
Dimension 5 - Test 1: Retrieval Relevance
===========================================
Tests whether the Researcher's search queries and relevance filtering
produce filtered results that are actually relevant to the research question.

Requires network access (DuckDuckGo web search).

Benchmark level:
  - module-level
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


RELEVANCE_JUDGE = """
You are evaluating whether a search result is relevant to a research question.

RESEARCH QUESTION: "{question}"
SEARCH QUERY USED: "{query}"

SEARCH RESULT TITLE: "{title}"
SEARCH RESULT TEXT: "{text}"

Evaluate:
1. **Topical relevance**: Is the result about a topic that the research question addresses?
2. **Informational value**: Does it contain information that helps answer the question?
3. **Specificity**: Is the information specific enough to be useful (vs generic)?

IMPORTANT: For cross-domain questions (e.g. "How does X in biology compare to Y
in machine learning?"), a result that covers EITHER domain deeply IS relevant —
it provides half the information needed for comparison. Only mark as irrelevant
if the result covers neither domain or is too superficial to be useful.

Score:
- relevance: 1 to 7 (overall relevance to the research question)
- informational_value: 1 to 7

Respond EXACTLY in JSON:
{{
  "relevant": true or false,
  "relevance": 1 to 7,
  "informational_value": 1 to 7,
  "reasoning": "one sentence"
}}
"""


QUESTIONS = [
    "How does synaptic consolidation during sleep relate to experience replay in reinforcement learning?",
    "What are the mathematical connections between genetic algorithms and gradient descent optimization?",
    "How does horizontal gene transfer in bacteria compare to knowledge distillation in machine learning?",
    "What mechanisms prevent catastrophic forgetting in biological neural circuits?",
    "How does the error-correcting structure of the genetic code relate to information theory?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument("--skip-network", action="store_true", default=False)
    parser.add_argument(
        "--out",
        default="benchmark/dim5/results/d5_retrieval_relevance.json",
    )
    args = parser.parse_args()

    if args.skip_network:
        print("TEST 1: Retrieval Relevance — SKIPPED (--skip-network)")
        report = {
            "test": "D5 - Retrieval Relevance",
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
    print("TEST 1: Retrieval Relevance (live web search)")
    print("=" * 60)

    researcher, _, _, _ = make_researcher(depth="standard")

    evaluations = []
    total_precision = 0.0
    total_relevance = 0.0
    total_raw_results = 0
    total_filtered_results = 0
    questions_with_filtered_results = 0

    for question in QUESTIONS:
        print(f"\n  Question: {question[:70]}...")

        queries = researcher._generate_queries(question, n=2)
        print(f"    Queries: {queries}")

        question_results = []
        query_breakdown = []
        relevant_count = 0
        raw_results_count = 0
        filtered_results_count = 0

        for query in queries:
            raw_results = researcher._web_search(query)
            filtered_results = researcher._filter_relevant(question, raw_results)
            raw_results_count += len(raw_results)
            filtered_results_count += len(filtered_results)
            total_raw_results += len(raw_results)

            print(
                f"    [{query[:30]}...] → raw={len(raw_results)} "
                f"filtered={len(filtered_results)}"
            )
            query_breakdown.append({
                "query": query,
                "raw_results_count": len(raw_results),
                "filtered_results_count": len(filtered_results),
            })

            for title, text, url in filtered_results:
                total_filtered_results += 1

                judgment = judge_json(
                    RELEVANCE_JUDGE.format(
                        question=question,
                        query=query,
                        title=title[:200],
                        text=text[:500],
                    ),
                    model=args.judge_model,
                    default={
                        "relevant": False, "relevance": 1,
                        "informational_value": 1, "reasoning": "Judge parse failed",
                    },
                )

                is_relevant = bool(judgment.get("relevant", False))
                relevance = float(judgment.get("relevance", 1))
                info_value = float(judgment.get("informational_value", 1))

                if is_relevant:
                    relevant_count += 1
                total_relevance += relevance

                question_results.append({
                    "query": query,
                    "title": title[:200],
                    "text_preview": text[:200],
                    "url": url,
                    "relevant": is_relevant,
                    "relevance": relevance,
                    "informational_value": info_value,
                    "judgment": judgment,
                })

                status = "✓" if is_relevant else "✗"
                print(f"      {status} [{relevance:.0f}/7] {title[:60]}")

            time.sleep(2)  # rate limiting

        if filtered_results_count > 0:
            questions_with_filtered_results += 1

        question_total = max(filtered_results_count, 1)
        precision = relevant_count / question_total
        total_precision += precision

        evaluations.append({
            "question": question,
            "queries": queries,
            "query_breakdown": query_breakdown,
            "raw_results_count": raw_results_count,
            "filtered_results_count": filtered_results_count,
            "relevant_count": relevant_count,
            "precision": round(precision, 3),
            "results": question_results,
        })

    n = max(len(QUESTIONS), 1)
    mean_precision = total_precision / n
    mean_relevance = total_relevance / max(total_filtered_results, 1)
    question_coverage = questions_with_filtered_results / n

    passed = (
        mean_precision >= 0.70 and
        mean_relevance >= 4.5 and
        question_coverage >= 0.60
    )

    report = {
        "test": "D5 - Retrieval Relevance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model, "skip_network": False},
        "summary": {
            "questions_evaluated": len(QUESTIONS),
            "total_raw_results": total_raw_results,
            "total_filtered_results": total_filtered_results,
            "questions_with_filtered_results": questions_with_filtered_results,
            "question_coverage": round(question_coverage, 3),
            "mean_precision": round(mean_precision, 3),
            "mean_relevance": round(mean_relevance, 3),
            "PASS": passed,
        },
        "evaluations": evaluations,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nMean precision    : {mean_precision:.2%}")
    print(f"Mean relevance    : {mean_relevance:.2f}/7")
    print(f"Question coverage : {question_coverage:.2%}")
    verdict = "PASS" if passed else "FAIL"
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
