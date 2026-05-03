"""
Dimension 1 — Test 2: Concept Coverage
======================================
Measures whether ingestion captures the major concepts from each benchmark
article, rather than producing only a few high-quality but incomplete nodes.

This benchmark uses the real `Ingestor.ingest(...)` path and then asks an
LLM judge whether each target concept is materially represented in the
resulting node set for that article.

Important benchmark note:
- We do NOT score against the full theoretical Wikipedia article while only
  ingesting the first truncated chunk.
- Instead, we fetch the full article extract and build a compact benchmark
  source from the lead plus windows around each target concept. That keeps
  the test fair: every scored concept is actually present in the source text
  given to the ingestor.

Pass criterion:
  - mean concept coverage across articles >= 70%
  - no article may fall below 50% concept coverage
  - all corpus articles must successfully yield nodes

Benchmark level:
  - module-level

Usage:
    python benchmark/dim1/test_d1_concept_coverage.py \
        --judge-model <ollama-model-name> \
        --out benchmark/dim1/results/d1_concept_coverage.json
"""

import os
import sys
import json
import time
import argparse
import statistics
import re
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


CORPUS = [
    {
        "id": "dna",
        "title": "DNA",
        "domain": "biology",
        "key_concepts": [
            "double helix", "base pairing", "replication",
            "transcription", "genetic code"
        ]
    },
    {
        "id": "thermodynamics",
        "title": "Thermodynamics",
        "domain": "physics",
        "key_concepts": [
            "entropy", "laws of thermodynamics", "heat", "work", "free energy"
        ]
    },
    {
        "id": "natural_selection",
        "title": "Natural Selection",
        "domain": "evolutionary_biology",
        "key_concepts": [
            "variation", "fitness", "adaptation", "selection pressure",
            "heritability"
        ]
    },
    {
        "id": "neural_network",
        "title": "Artificial neural network",
        "domain": "computer_science",
        "key_concepts": [
            "weights", "activation function", "backpropagation",
            "gradient descent", "overfitting"
        ]
    },
    {
        "id": "game_theory",
        "title": "Game theory",
        "domain": "economics",
        "key_concepts": [
            "Nash equilibrium", "payoff matrix", "prisoner's dilemma",
            "dominant strategy", "zero-sum"
        ]
    },
]


CONCEPT_COVERAGE_PROMPT = """You are evaluating whether an ingested node set
actually covers a target concept from a source article.

Article title: {title}
Article domain: {domain}
Target concept: {concept}

Extracted nodes:
{nodes}

Decide whether at least one extracted node materially captures the target
concept or a tight scientific paraphrase of it.

Rules:
- Count as covered if a node clearly states the concept itself, or explains its
  role/mechanism so directly that a scientist would say the concept is present.
- Do NOT count mere topical overlap.
- Do NOT count vague neighboring ideas.

Respond with JSON:
{{
  "covered": true or false,
  "matching_node": "<best matching node, or null>",
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence"
}}
Respond ONLY with JSON.
"""


@contextmanager
def override_models(model: str | None = None):
    from config import MODELS

    attrs = ("CREATIVE", "PRECISE", "REASONING", "CRITIC")
    old = {attr: getattr(MODELS, attr) for attr in attrs}
    try:
        if model:
            MODELS.CREATIVE = model
            MODELS.PRECISE = model
            MODELS.REASONING = model
            MODELS.CRITIC = model
        yield
    finally:
        for attr, value in old.items():
            setattr(MODELS, attr, value)


def fetch_wikipedia(title: str) -> str:
    import requests

    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "format": "json",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
    }
    resp = requests.get(
        api,
        params=params,
        timeout=20,
        headers={"User-Agent": "AutoScientist-Benchmark/1.0"},
    )
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""


def _normalize_for_match(text: str) -> str:
    return text.lower().replace("’", "'")


def _find_concept_spans(text: str, concepts: list[str]) -> dict[str, list[tuple[int, int]]]:
    normalized_text = _normalize_for_match(text)
    spans_by_concept = {}
    for concept in concepts:
        pattern = re.escape(_normalize_for_match(concept))
        matches = [
            (m.start(), m.end())
            for m in re.finditer(pattern, normalized_text)
        ]
        spans_by_concept[concept] = matches
    return spans_by_concept


def build_article_source(text: str, concepts: list[str],
                         lead_chars: int = 1800,
                         window_chars: int = 650,
                         max_chars: int = 14000) -> tuple[str, dict]:
    """
    Build a compact but concept-faithful source text from the full article.

    We keep the lead section, then add local windows around each target concept
    occurrence so the benchmark only judges concepts that were actually present
    in the ingested source.
    """
    if not text:
        return "", {
            "source_length": 0,
            "concepts_present_in_source": [],
            "concepts_missing_from_full_text": concepts[:],
            "segments_used": [],
        }

    concept_spans = _find_concept_spans(text, concepts)
    source_ranges = [(0, min(len(text), lead_chars))]

    concepts_present = []
    concepts_missing = []
    for concept in concepts:
        spans = concept_spans.get(concept, [])
        if not spans:
            concepts_missing.append(concept)
            continue
        concepts_present.append(concept)
        for start, end in spans[:2]:
            source_ranges.append((
                max(0, start - window_chars),
                min(len(text), end + window_chars),
            ))

    source_ranges.sort()
    merged = []
    for start, end in source_ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    pieces = []
    used_segments = []
    total_chars = 0
    for start, end in merged:
        segment = text[start:end].strip()
        if not segment:
            continue
        extra = len(segment) + (6 if pieces else 0)
        if total_chars + extra > max_chars:
            remaining = max_chars - total_chars - (6 if pieces else 0)
            if remaining <= 0:
                break
            segment = segment[:remaining].rstrip()
            end = start + len(segment)
        if pieces:
            pieces.append("\n\n...\n\n")
            total_chars += 6
        pieces.append(segment)
        used_segments.append({
            "start": start,
            "end": end,
            "length": len(segment),
        })
        total_chars += len(segment)
        if total_chars >= max_chars:
            break

    return "".join(pieces), {
        "source_length": total_chars,
        "concepts_present_in_source": concepts_present,
        "concepts_missing_from_full_text": concepts_missing,
        "segments_used": used_segments,
    }


def ingest_article(brain, ingestor, article: dict) -> dict:
    print(f"  Fetching: {article['title']}...")
    full_text = fetch_wikipedia(article["title"])
    if not full_text:
        print(f"    WARNING: empty text for {article['title']}")
        return {
            "text_found": False,
            "source_text": "",
            "source_meta": {
                "source_length": 0,
                "concepts_present_in_source": [],
                "concepts_missing_from_full_text": article["key_concepts"][:],
                "segments_used": [],
            },
            "nodes": [],
        }

    text, source_meta = build_article_source(full_text, article["key_concepts"])
    if not text:
        print(f"    WARNING: no benchmark source built for {article['title']}")
        return {
            "text_found": True,
            "source_text": "",
            "source_meta": source_meta,
            "nodes": [],
        }

    from graph.brain import EdgeSource

    new_ids = ingestor.ingest(text, source=EdgeSource.READING) or []
    nodes = []
    seen_ids = set()
    for nid in new_ids:
        if nid in seen_ids:
            continue
        node = brain.get_node(nid)
        if not node:
            continue
        seen_ids.add(nid)
        nodes.append({
            "node_id": nid,
            "statement": node["statement"],
            "node_type": node.get("node_type", "concept"),
            "cluster": node.get("cluster", "unknown"),
        })

    print(
        f"    Built {source_meta['source_length']} chars of benchmark source "
        f"covering {len(source_meta['concepts_present_in_source'])}/"
        f"{len(article['key_concepts'])} target concepts"
    )
    print(f"    Extracted {len(nodes)} nodes")
    return {
        "text_found": True,
        "source_text": text,
        "source_meta": source_meta,
        "nodes": nodes,
    }


def judge_concept_coverage(article: dict, concept: str, nodes: list, model: str) -> dict:
    from llm_utils import llm_call, require_json

    node_block = "\n".join(
        f"- {node['statement']}" for node in nodes
    ) if nodes else "- <no extracted nodes>"
    prompt = CONCEPT_COVERAGE_PROMPT.format(
        title=article["title"],
        domain=article["domain"],
        concept=concept,
        nodes=node_block,
    )
    raw = llm_call(prompt, temperature=0.0, model=model, role="precise")
    result = require_json(raw, default={})
    if "covered" not in result:
        return {
            "covered": False,
            "matching_node": None,
            "confidence": 0.0,
            "reasoning": "Judge parse failed",
            "parse_error": True,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="llama3.1:70b")
    parser.add_argument(
        "--out",
        default="benchmark/dim1/results/d1_concept_coverage.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from graph.brain import Brain
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)

    article_nodes = {}
    article_stats = {}

    with override_models(args.judge_model):
        print("=" * 60)
        print("PHASE 1: Ingesting benchmark corpus for coverage analysis")
        print("=" * 60)
        for article in CORPUS:
            result = ingest_article(brain, ingestor, article)
            article_nodes[article["id"]] = result["nodes"]
            article_stats[article["id"]] = {
                "text_found": result["text_found"],
                "source_length": result["source_meta"]["source_length"],
                "node_count": len(result["nodes"]),
                "source_concepts_present": result["source_meta"]["concepts_present_in_source"],
                "source_concepts_missing": result["source_meta"]["concepts_missing_from_full_text"],
                "source_segments": result["source_meta"]["segments_used"],
            }
            time.sleep(1)

        print("\n" + "=" * 60)
        print("PHASE 2: Judging concept coverage")
        print("=" * 60)
        concept_judgments = []
        skipped_concepts = []
        for article in CORPUS:
            nodes = article_nodes[article["id"]]
            source_concepts_present = set(
                article_stats[article["id"]]["source_concepts_present"]
            )
            for concept in article["key_concepts"]:
                if concept not in source_concepts_present:
                    skipped_concepts.append({
                        "article_id": article["id"],
                        "article_title": article["title"],
                        "concept": concept,
                        "reason": "concept not present in benchmark source text",
                    })
                    continue
                print(f"  [{article['id']}] {concept}")
                judgment = judge_concept_coverage(
                    article, concept, nodes, args.judge_model
                )
                concept_judgments.append({
                    "article_id": article["id"],
                    "article_title": article["title"],
                    "concept": concept,
                    "judgment": judgment,
                })
                time.sleep(0.2)

    per_article = {}
    coverage_scores = []
    parse_errors = sum(
        1 for j in concept_judgments if j["judgment"].get("parse_error")
    )
    source_aligned_concepts = sum(
        len(article_stats[article["id"]]["source_concepts_present"])
        for article in CORPUS
    )
    articles_with_text = sum(
        1 for article in CORPUS if article_stats[article["id"]]["text_found"]
    )
    articles_with_nodes = sum(
        1 for article in CORPUS if article_stats[article["id"]]["node_count"] > 0
    )

    for article in CORPUS:
        judgments = [
            j for j in concept_judgments if j["article_id"] == article["id"]
        ]
        covered = [j for j in judgments if j["judgment"].get("covered")]
        evaluable_concepts = len(article_stats[article["id"]]["source_concepts_present"])
        coverage = len(covered) / evaluable_concepts if evaluable_concepts else 0.0
        coverage_scores.append(coverage)
        per_article[article["id"]] = {
            "title": article["title"],
            "domain": article["domain"],
            "text_found": article_stats[article["id"]]["text_found"],
            "source_length": article_stats[article["id"]]["source_length"],
            "node_count": article_stats[article["id"]]["node_count"],
            "source_concepts_present": article_stats[article["id"]]["source_concepts_present"],
            "source_concepts_missing": article_stats[article["id"]]["source_concepts_missing"],
            "concepts_total": len(article["key_concepts"]),
            "concepts_evaluable": evaluable_concepts,
            "concepts_covered": len(covered),
            "coverage_fraction": round(coverage, 3),
            "missed_concepts": [
                j["concept"] for j in judgments if not j["judgment"].get("covered")
            ],
            "skipped_concepts": [
                s["concept"] for s in skipped_concepts if s["article_id"] == article["id"]
            ],
        }

    mean_coverage = statistics.mean(coverage_scores) if coverage_scores else 0.0
    min_article_coverage = min(coverage_scores) if coverage_scores else 0.0
    total_concepts = sum(len(article["key_concepts"]) for article in CORPUS)
    evaluable_concepts_total = len(concept_judgments)
    covered_concepts = sum(
        1 for j in concept_judgments if j["judgment"].get("covered")
    )

    passed = (
        articles_with_text == len(CORPUS) and
        articles_with_nodes == len(CORPUS) and
        evaluable_concepts_total > 0 and
        mean_coverage >= 0.70 and
        min_article_coverage >= 0.50
    )

    report = {
        "test": "D1 — Concept Coverage",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "judge_model": args.judge_model,
            "corpus_size": len(CORPUS),
            "total_concepts": total_concepts,
        },
        "summary": {
            "articles_with_text": articles_with_text,
            "articles_with_nodes": articles_with_nodes,
            "source_aligned_concepts": source_aligned_concepts,
            "evaluable_concepts": evaluable_concepts_total,
            "covered_concepts": covered_concepts,
            "total_concepts": total_concepts,
            "mean_concept_coverage": round(mean_coverage, 3),
            "min_article_coverage": round(min_article_coverage, 3),
            "parse_errors": parse_errors,
            "PASS_corpus_coverage": (
                articles_with_text == len(CORPUS) and
                articles_with_nodes == len(CORPUS)
            ),
            "PASS_mean_concept_coverage": mean_coverage >= 0.70,
            "PASS_min_article_coverage": min_article_coverage >= 0.50,
            "PASS": passed,
            "pass_threshold_mean_concept_coverage": 0.70,
            "pass_threshold_min_article_coverage": 0.50,
        },
        "per_article": per_article,
        "skipped_concepts": skipped_concepts,
        "concept_judgments": concept_judgments,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS — D1: Concept Coverage")
    print("=" * 60)
    print(f"Articles with text      : {articles_with_text}/{len(CORPUS)}")
    print(f"Articles with nodes     : {articles_with_nodes}/{len(CORPUS)}")
    print(f"Source-aligned concepts : {source_aligned_concepts}/{total_concepts}")
    print(f"Covered concepts        : {covered_concepts}/{evaluable_concepts_total}")
    print(f"Mean concept coverage   : {mean_coverage:.1%} (threshold: 70%)")
    print(f"Min article coverage    : {min_article_coverage:.1%} (threshold: 50%)")
    print(f"Parse errors            : {parse_errors}")
    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"\nOVERALL VERDICT: {verdict}")
    print(f"Full report saved to: {args.out}")


if __name__ == "__main__":
    main()
