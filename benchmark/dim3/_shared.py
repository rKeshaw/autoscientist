import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MISSION = (
    "How do biological and artificial learning systems balance "
    "exploration with stability?"
)

CORPUS = [
    {"id": "dna", "title": "DNA"},
    {"id": "thermodynamics", "title": "Thermodynamics"},
    {"id": "natural_selection", "title": "Natural selection"},
    {"id": "ann", "title": "Artificial neural network"},
    {"id": "game_theory", "title": "Game theory"},
    {"id": "genetics", "title": "Genetics"},
    {"id": "epigenetics", "title": "Epigenetics"},
    {"id": "mutation", "title": "Mutation"},
]

SUPPORTED_PATTERNS = {
    "dialectical",
    "analogical",
    "reductive",
    "experimental",
    "integrative",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "how", "if", "in", "into", "is", "it", "of", "on",
    "or", "should", "than", "that", "the", "to", "what", "when", "where",
    "which", "while", "with", "would",
}


def fetch_wikipedia(title: str) -> str:
    import requests

    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "format": "json",
    }
    resp = requests.get(
        api,
        params=params,
        timeout=20,
        headers={"User-Agent": "AutoScientist-Benchmark/3.0"},
    )
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")[:8000]
    return ""


def _shared_paths(dim: str = "dim3"):
    shared_dir = ROOT / "benchmark" / dim / "shared"
    return shared_dir, shared_dir / "brain.json", shared_dir / "embedding_index"


def prepare_shared_graph(dim: str = "dim3"):
    from graph.brain import Brain, EdgeSource
    from embedding_index import EmbeddingIndex
    from ingestion.ingestor import Ingestor
    from observer.observer import Observer

    shared_dir, brain_path, index_path = _shared_paths(dim)
    os.makedirs(shared_dir, exist_ok=True)

    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)

    for article in CORPUS:
        print(f"  Ingesting: {article['title']}...")
        text = fetch_wikipedia(article["title"])
        if text:
            ingestor.ingest(text, source=EdgeSource.READING)
            time.sleep(1)

    brain.save(brain_path)
    emb_index.save(index_path)
    return brain, emb_index


def load_or_build_shared_graph():
    from graph.brain import Brain
    from embedding_index import EmbeddingIndex

    dim3_dir, dim3_brain_path, dim3_index_path = _shared_paths("dim3")
    dim2_dir, dim2_brain_path, dim2_index_path = _shared_paths("dim2")

    brain = Brain()

    if dim3_brain_path.exists() and Path(str(dim3_index_path) + ".json").exists():
        brain.load(str(dim3_brain_path))
        emb_index = EmbeddingIndex.load(str(dim3_index_path))
        return brain, emb_index

    if dim2_brain_path.exists() and Path(str(dim2_index_path) + ".json").exists():
        brain.load(str(dim2_brain_path))
        emb_index = EmbeddingIndex.load(str(dim2_index_path))
        return brain, emb_index

    print("  Shared graph not found. Building Dim 3 shared graph from scratch...")
    return prepare_shared_graph("dim3")


def isolate_policy(tag: str):
    from thinker.policy import CognitivePolicy

    policy_dir = ROOT / "benchmark" / "dim3" / "tmp" / "policy"
    os.makedirs(policy_dir, exist_ok=True)
    policy_path = policy_dir / f"_policy_{tag}.json"
    if policy_path.exists():
        policy_path.unlink()
    CognitivePolicy.POLICY_PATH = str(policy_path)
    return policy_path


def make_thinker(
    mission: str = DEFAULT_MISSION,
    policy_tag: str = "default",
    critic=None,
):
    from observer.observer import Observer
    from thinker.thinker import Thinker

    isolate_policy(policy_tag)
    brain, emb_index = load_or_build_shared_graph()
    observer = Observer(brain)
    if mission:
        brain.set_mission(mission)
        brain.complete_transition()
    thinker = Thinker(
        brain,
        observer=observer,
        embedding_index=emb_index,
        critic=critic,
    )
    thinker.policy.epsilon = 0.0
    return thinker, brain, observer, emb_index


def add_focus_question(
    brain,
    emb_index,
    text: str,
    node_type,
    cluster: str = "thinking",
    importance: float = 0.8,
):
    from graph.brain import Node, NodeStatus
    from embedding import embed as shared_embed

    node = Node(
        statement=text,
        node_type=node_type,
        cluster=cluster,
        status=NodeStatus.UNCERTAIN,
        importance=importance,
        source_quality=0.7,
    )
    nid = brain.add_node(node)
    if emb_index:
        emb_index.add(nid, shared_embed(text))
    brain.focus_on(nid)
    return nid


def judge_json(prompt: str, model: str, default: dict):
    from llm_utils import llm_call, require_json

    json_system = (
        "You are a strict evaluator. Respond ONLY with a valid JSON object. "
        "Do not include prose outside JSON, markdown fences, or comments."
    )
    raw = llm_call(
        prompt,
        temperature=0.1,
        model=model,
        system=json_system,
        role="precise",
    )
    result = require_json(raw, default=None)
    if isinstance(result, dict):
        result.setdefault("_parse_failed", False)
        return result

    repair_prompt = (
        "Repair the following evaluator output into valid JSON matching this "
        f"schema:\n{json.dumps(default, indent=2)}\n\n"
        f"Evaluator output:\n{raw}\n\n"
        "Return ONLY the repaired JSON object."
    )
    repaired_raw = llm_call(
        repair_prompt,
        temperature=0.0,
        model=model,
        system=json_system,
        role="precise",
    )
    repaired = require_json(repaired_raw, default=None)
    if isinstance(repaired, dict):
        repaired.setdefault("_parse_failed", False)
        repaired["_repaired_from_raw"] = True
        return repaired

    fallback = dict(default)
    fallback["_parse_failed"] = True
    fallback["_raw_response"] = raw[:1000]
    return fallback


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def content_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {token for token in tokens if token not in STOPWORDS}


def unique_fraction(texts: list[str]) -> float:
    cleaned = [normalize_text(text) for text in texts if normalize_text(text)]
    if not cleaned:
        return 0.0
    return len(set(cleaned)) / len(cleaned)


def max_restatement_ratio(main_question: str, derived_questions: list[str]) -> float:
    main_tokens = content_tokens(main_question)
    if not main_tokens or not derived_questions:
        return 0.0

    ratios = []
    for item in derived_questions:
        derived_tokens = content_tokens(item)
        if not derived_tokens:
            continue
        overlap = len(main_tokens & derived_tokens) / max(len(derived_tokens), 1)
        ratios.append(overlap)
    return max(ratios) if ratios else 0.0
