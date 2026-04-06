<p align="center">
  <h1 align="center">🧠 AutoScientist</h1>
  <p align="center">
    <em>An autonomous research agent that thinks, reads, dreams, and writes — modeled after the cognitive rhythms of a human scientist.</em>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#modules">Modules</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#license">License</a>
  </p>
</p>

---

AutoScientist is an **autonomous scientific research system** modeled after biologically-inspired cognitive architectures. It builds and maintains a knowledge graph by reading papers, forming top-down expectations, dreaming about structural analogies, and learning optimal reasoning strategies via reinforcement learning — all regulated by a dynamic Salience Network.

The system moves beyond static pipelines into a **neuro-active architecture**: predictive processing (Active Inference), procedural memory (Contextual Bandits), and episodic trace replay (Hippocampus) are all modulated by synthetic Dopamine and Frustration signals.

## Features

| | |
|---|---|
| 🧠 **Neuro-Modulation** | Dopamine & Frustration signals regulate mission persistence and task priority |
| 🔮 **Predictive Processing** | Top-down expectations (Active Inference) calculate "Surprise" to modulate node importance |
| 🕹️ **Procedural RL** | A Contextual Bandit learns the best reasoning patterns (Dialectical, Analogical, etc.) per domain |
| 🎞️ **Episodic Memory** | Chronological event strips allow for "Hippocampal Replay" during dream cycles |
| ⚡ **Salience Network** | Priority-queue scheduler that dynamically interrupts routine tasks for urgent breakthroughs |
| 🌙 **Dream Cycles** | Nightly graph walks and episodic re-processing to find structural isomorphisms |
| 🧪 **Computational Sandbox** | Auto-generates and runs Python experiments with frustration-based retry loops |
| 🛡️ **System 2 Critic** | High-stakes claims must survive adversarial multi-turn dialogue before graph entry |

## Quickstart

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** running locally with a model pulled (default: `mixtral:latest`)

### Installation

```bash
git clone https://github.com/yourusername/autoscientist.git
cd autoscientist
pip install -r requirements.txt
```

### Bootstrap a Research Brain

```bash
python bootstrap.py "How does sleep contribute to creative problem-solving?"
```

### Launch the Web UI

```bash
python gui/app.py
# Open http://localhost:5000
```

### Start the Autonomous Scheduler

```bash
python -m scheduler.scheduler
```

Run a single phase manually:

```bash
python -m scheduler.scheduler --mode dream
python -m scheduler.scheduler --mode thinking
python -m scheduler.scheduler --mode cycle   # full cycle immediately
```

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     AutoScientist                         │
│                                                           │
│  Predictive Reader → Ingestor (Surprise) → Brain (Graph)  │
│                                              │            │
│        ┌───────────────────┬─────────────────┴──┐         │
│        │                   │                    │         │
│     Dreamer             Thinker            Researcher     │
│   (REM Replay)      (Procedural RL)                       │
│        │                   │                              │
│        └─────────┬─────────┘                              │
│                  │                                        │
│           Salience Network (Priority Queue Loop)          │
│                  │                                        │
│       Critic (System 2) ←───→ Neuromodulators             │
│    (Dopamine/Frustration)      (Dynamic Gating)           │
└───────────────────────────────────────────────────────────┘
```

Instead of a rigid clock, AutoScientist uses a **Salience Network** that prioritizes tasks based on cognitive load and neuromodulation:

| Priority | Task | Trigger / Condition |
|----------|------|-----------|
| **URGENT** | 💬 Conversation / 🚨 Sandbox Failure | Direct user input or critical experimental collapse |
| **HIGH** | 🤔 Thinking / 🔬 Discovery | Triggered by **Dopamine Spikes** after a mission advance |
| **ROUTINE** | 📖 Reading / ✍️ Writing | Default daytime cycle for incremental knowledge gain |
| **BACKGROUND** | 🌙 Dreaming / 🔧 Consolidation | Triggered by low dopamine or high frustration (Reset) |

## Modules

### Core

| Module | Purpose |
|--------|---------|
| `graph/brain.py` | Knowledge graph — nodes, edges, mission, working memory |
| `embedding_index.py` | FAISS-backed vector index |
| `llm_utils.py` | Unified LLM interface — role-based model selection, JSON parsing |
| `config.py` | Thresholds, per-role model config, Critic config |
| `insight_buffer.py` | Delayed insight mechanism — buffers near-miss pairs for re-evaluation |
| `persistence.py` | Atomic JSON writes |

### Cognitive

| Module | Purpose |
|--------|---------|
| `graph/brain.py` | Knowledge graph + Neuromodulator state (Dopamine, Frustration) |
| `graph/episodic.py` | **EpisodicStrip** — chronological record of cognitive events |
| `thinker/policy.py` | **Procedural Policy** — Contextual Bandit for RL strategy selection |
| `scheduler/scheduler.py` | **Salience Network** — Priority-queue based event loop |
| `ingestion/ingestor.py` | **Predictive Ingestor** — surprises/prediction-error evaluation |
| `dreamer/` | **REM/NREM Dreamer** — with Hippocampal Replay integration |
| `critic/` | System 2 — adversarial gating and frustration feedback |
| `sandbox/` | Computational hypothesis testing with frustration retry loops |

## Configuration

### Model Selection (`config.py`)

Role-based model routing — different tasks can use different models:

```python
class ModelConfig:
    CREATIVE     = "mixtral:latest"   # Dreaming, synthesis
    PRECISE      = "mixtral:latest"   # JSON extraction
    CODE         = "mixtral:latest"   # Sandbox experiments
    REASONING    = "mixtral:latest"   # Thinker
    CONVERSATION = "mixtral:latest"   # Chat
    CRITIC       = "mixtral:latest"   # System 2 adversarial review
```

Swap models per role as needed:
```python
MODELS.CRITIC    = "llama3.1:70b"   # more rigorous System 2
MODELS.CREATIVE  = "llama3.1:70b"   # better dreaming
```

### Thresholds (`config.py`)

```python
class ThresholdConfig:
    MERGE_NODE      = 0.72   # Cosine similarity to merge near-duplicate nodes
    DUPLICATE_MERGE = 0.88   # Strict duplicate detection
    WEAK_EDGE       = 0.58   # Minimum similarity for associative edges
    COHERENCE       = 0.65   # Cross-domain insight quality threshold
    GAP_CONFIDENCE  = 0.75   # Confidence needed to infer gap nodes
```

### Critic / System 2 (`config.py`)

```python
class CriticConfig:
    ACTIVATION_THRESHOLD    = 0.65   # Minimum importance to trigger review
    MAX_DIALOGUE_TURNS      = 3      # Adversarial rounds
    ACCEPT_CONFIDENCE_FLOOR = 0.50   # Minimum confidence to ACCEPT
    ALWAYS_REVIEW_TYPES     = ["hypothesis", "synthesis",
                               "structural_analogy", "deep_isomorphism"]
    BYPASS_TYPES            = ["concept", "associated", "surface_analogy"]
    MAX_REFINE_ITERATIONS   = 2
```

Critic verdicts: **ACCEPT** · **REFINE** · **REJECT** · **DEFER** (→ InsightBuffer)

## Knowledge Graph

### Node Types

`concept` · `hypothesis` · `question` · `answer` · `synthesis` · `gap` · `mission` · `empirical`

### Edge Types

`supports` · `causes` · `contradicts` · `surface_analogy` · `structural_analogy` · `deep_isomorphism` · `associated`

## Data & Logs

```
data/
├── brain.json              # Knowledge graph state
├── observer.json           # Observer state and agenda
├── embedding_index/        # FAISS index
├── insight_buffer.json     # Pending near-miss pairs
└── daily_new_nodes.json    # Daily node ledger

logs/
├── cycle_log.json
├── research_log.json
├── notebook.json
└── sandbox_results.json
```

## Running Tests

```bash
pytest tests/ -v
```

## License

[GNU General Public License v3.0](LICENSE)

---

<p align="center">
  <em>"The mind, once stretched by a new idea, never returns to its original dimensions."</em><br/>
  — Oliver Wendell Holmes Sr.
</p>
