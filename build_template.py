"""
Template Brain Builder — Pre-build foundational knowledge brains.

Creates reusable brain templates seeded with large corpora of foundational
knowledge that new research missions can fork from.

Usage:
    python build_template.py --name general_scientist --manifest manifests/general.json
    python build_template.py --name physics_researcher --manifest manifests/physics.json

Templates are stored in brain_templates/ and loaded via:
    python bootstrap.py --template general_scientist "Your mission question"
"""

import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from graph.brain import Brain, BrainMode
from observer.observer import Observer
from ingestion.ingestor import Ingestor
from notebook.notebook import Notebook
from reader.reader import Reader
from embedding_index import EmbeddingIndex
from embedding import embed as shared_embed
from llm_utils import llm_json

TEMPLATE_DIR = "brain_templates"
MANIFEST_DIR = "brain_templates/manifests"

# ── Default manifests ─────────────────────────────────────────────────────────

GENERAL_MANIFEST = {
    "name": "general_scientist",
    "description": "Foundational knowledge for an educated scientist",
    "domains": [
        {
            "domain": "Scientific Method & Epistemology",
            "sources": [
                "https://en.wikipedia.org/wiki/Scientific_method",
                "https://en.wikipedia.org/wiki/Epistemology",
                "https://en.wikipedia.org/wiki/Falsifiability",
                "https://en.wikipedia.org/wiki/Paradigm_shift",
                "https://en.wikipedia.org/wiki/Peer_review",
            ]
        },
        {
            "domain": "Mathematics Foundations",
            "sources": [
                "https://en.wikipedia.org/wiki/Calculus",
                "https://en.wikipedia.org/wiki/Statistics",
                "https://en.wikipedia.org/wiki/Probability",
                "https://en.wikipedia.org/wiki/Logic",
                "https://en.wikipedia.org/wiki/Set_theory",
            ]
        },
        {
            "domain": "Physics Foundations",
            "sources": [
                "https://en.wikipedia.org/wiki/Classical_mechanics",
                "https://en.wikipedia.org/wiki/Thermodynamics",
                "https://en.wikipedia.org/wiki/Electromagnetism",
                "https://en.wikipedia.org/wiki/Quantum_mechanics",
                "https://en.wikipedia.org/wiki/General_relativity",
            ]
        },
        {
            "domain": "Biology Foundations",
            "sources": [
                "https://en.wikipedia.org/wiki/Evolution",
                "https://en.wikipedia.org/wiki/Cell_(biology)",
                "https://en.wikipedia.org/wiki/DNA",
                "https://en.wikipedia.org/wiki/Ecology",
            ]
        },
        {
            "domain": "Systems Thinking",
            "sources": [
                "https://en.wikipedia.org/wiki/Systems_theory",
                "https://en.wikipedia.org/wiki/Complexity",
                "https://en.wikipedia.org/wiki/Emergence",
                "https://en.wikipedia.org/wiki/Feedback",
            ]
        },
        {
            "domain": "History of Science",
            "sources": [
                "https://en.wikipedia.org/wiki/History_of_science",
                "https://en.wikipedia.org/wiki/Scientific_revolution",
                "https://en.wikipedia.org/wiki/Industrial_Revolution",
            ]
        }
    ]
}


def ensure_dirs():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)


def save_default_manifests():
    """Save built-in manifests to disk."""
    ensure_dirs()
    path = os.path.join(MANIFEST_DIR, "general.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(GENERAL_MANIFEST, f, indent=2)
        print(f"Saved default manifest: {path}")


def load_manifest(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_template(manifest: dict, max_per_domain: int = 5):
    """
    Build a template brain from a manifest.

    Args:
        manifest: dict with 'name', 'description', 'domains' (each with 'sources')
        max_per_domain: Max articles to absorb per domain
    """
    name = manifest["name"]
    description = manifest.get("description", "")

    print(f"\n{'='*60}")
    print(f"Building template brain: {name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    brain     = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer  = Observer(brain)
    notebook  = Notebook(brain, observer=observer)
    reader    = Reader(brain, observer=observer, notebook=notebook)
    ingestor  = Ingestor(brain, embedding_index=emb_index)

    total_absorbed = 0
    domains = manifest.get("domains", [])

    for di, domain_info in enumerate(domains):
        domain_name = domain_info["domain"]
        sources = domain_info.get("sources", [])

        print(f"\n[{di+1}/{len(domains)}] Domain: {domain_name}")
        print(f"  Sources: {len(sources)}")

        for si, url in enumerate(sources[:max_per_domain]):
            try:
                title = url.split("/wiki/")[-1].replace("_", " ") \
                        if "wikipedia" in url else url
                result = reader.absorb_url(url, title=title,
                                          source_type="wikipedia")
                if result.success:
                    total_absorbed += 1
                    print(f"  ✓ [{si+1}] {title} — {result.node_count} nodes")
                else:
                    print(f"  ✗ [{si+1}] {title} — {result.error}")
            except Exception as e:
                print(f"  ✗ [{si+1}] ERROR: {e}")
            time.sleep(1)

    # Save template
    ensure_dirs()
    brain_path = os.path.join(TEMPLATE_DIR, f"{name}.brain.json")
    index_path = os.path.join(TEMPLATE_DIR, f"{name}.index")

    brain.save(brain_path)
    emb_index.save(index_path)

    print(f"\n{'='*60}")
    print(f"Template '{name}' built successfully")
    print(f"  Absorbed: {total_absorbed} articles")
    print(f"  Brain: {brain.stats()['nodes']} nodes, "
          f"{brain.stats()['edges']} edges")
    print(f"  Saved: {brain_path}")
    print(f"{'='*60}")

    return brain.stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a DREAMER template brain")
    parser.add_argument("--name", default=None,
                        help="Template name (overrides manifest)")
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest JSON")
    parser.add_argument("--save-defaults", action="store_true",
                        help="Save default manifests to disk")
    parser.add_argument("--max-per-domain", type=int, default=5,
                        help="Max articles per domain")

    args = parser.parse_args()

    if args.save_defaults:
        save_default_manifests()

    if args.manifest:
        manifest = load_manifest(args.manifest)
    else:
        manifest = GENERAL_MANIFEST

    if args.name:
        manifest["name"] = args.name

    build_template(manifest, max_per_domain=args.max_per_domain)
