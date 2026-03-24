"""
DREAMER Bootstrap Script

Builds the initial knowledge graph from scratch:
1. Sets the central mission question
2. Seeds the reading list with foundational texts across key domains
3. Seeds the observer agenda with foundational questions
4. Runs initial absorption cycles
5. Runs one dream cycle and consolidation

Run once: python3 bootstrap.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from graph.brain import Brain, BrainMode
from observer.observer import Observer
from ingestion.ingestor import Ingestor, EdgeSource
from dreamer.dreamer import Dreamer, DreamMode
from consolidator.consolidator import Consolidator
from notebook.notebook import Notebook
from reader.reader import Reader

# ── Config ────────────────────────────────────────────────────────────────────

MISSION_QUESTION = (
    "Are there universal patterns in how scientific revolutions happen — "
    "and if so, what does the current state of knowledge suggest about "
    "when and where the next major revolution will occur?"
)

MISSION_CONTEXT = (
    "Starting from Kuhn's structure of scientific revolutions, "
    "asking whether the mechanism is universal, predictable, or acceleratable. "
    "Relevant domains: philosophy of science, history of science, "
    "complexity theory, sociology of knowledge, information theory."
)

# ── Seed reading list ─────────────────────────────────────────────────────────
# Curated Wikipedia articles spanning the key domains
# Each is a distinct cluster to ensure cross-domain density

SEED_ARTICLES = [
    # Philosophy of science — foundations
    {
        "url":   "https://en.wikipedia.org/wiki/The_Structure_of_Scientific_Revolutions",
        "title": "The Structure of Scientific Revolutions",
        "type":  "wikipedia",
        "reason": "Core text — Kuhn's paradigm shift theory"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Paradigm_shift",
        "title": "Paradigm shift",
        "type":  "wikipedia",
        "reason": "Central concept of the mission"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Philosophy_of_science",
        "title": "Philosophy of science",
        "type":  "wikipedia",
        "reason": "Broad overview of the domain"
    },
    # History of specific revolutions
    {
        "url":   "https://en.wikipedia.org/wiki/Copernican_Revolution",
        "title": "Copernican Revolution",
        "type":  "wikipedia",
        "reason": "Archetypal scientific revolution — astronomical"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Scientific_Revolution",
        "title": "Scientific Revolution",
        "type":  "wikipedia",
        "reason": "The 16th-17th century transformation of natural philosophy"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/History_of_quantum_mechanics",
        "title": "History of quantum mechanics",
        "type":  "wikipedia",
        "reason": "Modern paradigm shift in physics — rapid and contested"
    },
    # Competing philosophy of science frameworks
    {
        "url":   "https://en.wikipedia.org/wiki/Imre_Lakatos",
        "title": "Imre Lakatos — research programmes",
        "type":  "wikipedia",
        "reason": "Lakatos's alternative to Kuhn — progressive vs degenerative"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Karl_Popper",
        "title": "Karl Popper — falsificationism",
        "type":  "wikipedia",
        "reason": "Popper's account of scientific progress — contrasts with Kuhn"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Paul_Feyerabend",
        "title": "Paul Feyerabend — against method",
        "type":  "wikipedia",
        "reason": "Radical critique — science has no universal method"
    },
    # Complexity and emergence
    {
        "url":   "https://en.wikipedia.org/wiki/Complex_system",
        "title": "Complex systems",
        "type":  "wikipedia",
        "reason": "Framework for understanding revolutions as phase transitions"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Phase_transition",
        "title": "Phase transition",
        "type":  "wikipedia",
        "reason": "Physical analog — is a scientific revolution a phase transition?"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Punctuated_equilibrium",
        "title": "Punctuated equilibrium",
        "type":  "wikipedia",
        "reason": "Evolutionary analog to scientific revolution — rapid change after stasis"
    },
    # Network and information theory
    {
        "url":   "https://en.wikipedia.org/wiki/Network_science",
        "title": "Network science",
        "type":  "wikipedia",
        "reason": "Scientific communities as networks — structural analysis"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Information_theory",
        "title": "Information theory",
        "type":  "wikipedia",
        "reason": "Is a revolution a compression event? Shannon entropy framing"
    },
    # Sociology of science
    {
        "url":   "https://en.wikipedia.org/wiki/Sociology_of_scientific_knowledge",
        "title": "Sociology of scientific knowledge",
        "type":  "wikipedia",
        "reason": "Social construction of scientific facts — strong programme"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Merton_thesis",
        "title": "Merton thesis",
        "type":  "wikipedia",
        "reason": "Social conditions enabling scientific revolution"
    },
    # Mathematics and formalism
    {
        "url":   "https://en.wikipedia.org/wiki/Mathematical_universe_hypothesis",
        "title": "Mathematical universe hypothesis",
        "type":  "wikipedia",
        "reason": "Deep question about mathematics and physical reality"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Anomaly_(physics)",
        "title": "Anomaly in science",
        "type":  "wikipedia",
        "reason": "Anomaly accumulation — the engine of Kuhnian crisis"
    },
    # Psychology and cognition
    {
        "url":   "https://en.wikipedia.org/wiki/Gestalt_psychology",
        "title": "Gestalt psychology — perceptual shifts",
        "type":  "wikipedia",
        "reason": "Kuhn's duck-rabbit analogy — paradigm shift as perceptual flip"
    },
    {
        "url":   "https://en.wikipedia.org/wiki/Cognitive_revolution",
        "title": "Cognitive revolution",
        "type":  "wikipedia",
        "reason": "A recent scientific revolution — how it happened"
    },
]

# ── Seed questions ────────────────────────────────────────────────────────────

SEED_QUESTIONS = [
    "What distinguishes a paradigm shift from incremental scientific progress?",
    "Is scientific revolution more analogous to punctuated equilibrium or gradual phase transition?",
    "Why do major scientific revolutions often happen at domain boundaries?",
    "Is there a threshold of anomaly accumulation that reliably precedes a revolution?",
    "What role do outsiders and interdisciplinary thinkers play in scientific revolutions?",
    "Are there scientific revolutions that were suppressed, and what caused their suppression?",
    "What is the relationship between mathematical formalism and conceptual revolution?",
    "Do revolutions happen faster or slower as a field matures?",
    "Is there a universal structure to the preparatory period before a scientific revolution?",
    "Can the conditions for a scientific revolution be predicted from the current state of a field?",
    "What current fields in science show the anomaly patterns Kuhn described as pre-revolutionary?",
    "Is there a difference between how revolutions happen in physics vs biology vs social sciences?",
    "Does the internet and open access change how scientific revolutions propagate?",
    "What is the role of instrumentation and technology in enabling scientific revolutions?",
    "Is a scientific revolution a compression of knowledge — does it always simplify?",
    "Are there universal patterns in the resistance to scientific revolutions?",
    "What distinguishes a genuine paradigm shift from a failed revolution?",
    "Could there be a science of scientific revolutions — a meta-science?",
    "Does complexity theory predict anything about when the next major revolution will occur?",
    "What field is currently most overdue for a revolution based on anomaly accumulation?",
]

# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap():
    print("\n" + "="*60)
    print("DREAMER Bootstrap")
    print("="*60)

    # ensure data directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # initialise brain
    brain    = Brain()
    observer = Observer(brain)
    notebook = Notebook(brain, observer=observer)
    reader   = Reader(brain, observer=observer, notebook=notebook)
    ingestor = Ingestor(brain)

    # ── Step 1: Set mission ───────────────────────────────────────────────────
    print("\n[1/5] Setting mission...")
    brain.set_mission(MISSION_QUESTION, MISSION_CONTEXT)
    print(f"  Mode: {brain.get_mode()}")

    # ── Step 2: Seed observer agenda ──────────────────────────────────────────
    print(f"\n[2/5] Seeding agenda with {len(SEED_QUESTIONS)} questions...")
    for i, q in enumerate(SEED_QUESTIONS):
        item = observer.add_to_agenda(
            text      = q,
            item_type = "question",
            cycle     = 0,
            step      = i
        )
        item.priority = 0.7

    # ── Step 3: Seed reading list ─────────────────────────────────────────────
    print(f"\n[3/5] Seeding reading list with {len(SEED_ARTICLES)} articles...")
    for article in SEED_ARTICLES:
        reader.add_to_list(
            url         = article["url"],
            title       = article["title"],
            source_type = article["type"],
            priority    = 0.8,
            added_by    = "bootstrap",
            reason      = article["reason"]
        )
    print(f"  Reading list: {reader.stats()}")

    # ── Step 4: Absorb first batch ────────────────────────────────────────────
    print(f"\n[4/5] Absorbing first 8 articles...")
    print("  (This will take several minutes — each article runs LLM extraction)\n")

    absorbed = 0
    for entry in reader.get_unread(8):
        result = reader.absorb_entry(entry)
        if result.success:
            absorbed += 1
            print(f"  ✓ {result.title} — {result.node_count} nodes")
        else:
            print(f"  ✗ {entry.title} — {result.error}")
        time.sleep(2)

    print(f"\n  Absorbed {absorbed} articles")
    print(f"  Brain: {brain.stats()['nodes']} nodes, "
          f"{brain.stats()['edges']} edges, "
          f"{brain.stats()['clusters']} clusters")

    # ── Step 5: Initial dream + consolidation ─────────────────────────────────
    print("\n[5/5] Running initial dream cycle...")
    brain.complete_transition()   # move from TRANSITIONAL to FOCUSED

    dreamer      = Dreamer(brain, research_agenda=observer)
    consolidator = Consolidator(brain, observer=observer)

    # transitional dream — higher temperature, more steps
    log = dreamer.dream(
        mode        = DreamMode.WANDERING,
        steps       = 12,
        temperature = 0.8,    # higher for first chaotic cycle
        run_nrem    = True,
        log_path    = "logs/dream_bootstrap.json"
    )
    observer.observe(log)
    notebook.write_morning_entry(log, cycle=0)

    # consolidate
    report = consolidator.consolidate(save_path="logs/consolidation_bootstrap.json")
    notebook.write_evening_entry(report, cycle=0)
    notebook.update_running_hypothesis(cycle=0)

    # ── Save everything ───────────────────────────────────────────────────────
    brain.save("data/brain.json")
    observer.save("data/observer.json")
    notebook.save()

    print("\n" + "="*60)
    print("Bootstrap complete.")
    print(f"Brain: {brain.stats()['nodes']} nodes, "
          f"{brain.stats()['edges']} edges, "
          f"{brain.stats()['clusters']} clusters")
    print(f"Agenda: {len(observer.agenda)} questions")
    print(f"Reading list: {reader.stats()['unread']} articles still unread")
    print(f"Mode: {brain.get_mode()}")
    print(f"\nMission: {MISSION_QUESTION}")
    print("\nReady. Start the GUI with: python3 gui/app.py")
    print("="*60)

if __name__ == "__main__":
    bootstrap()
