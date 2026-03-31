<p align="center">
  <h1 align="center">🧠 AutoScientist</h1>
  <p align="center">
    <em>An autonomous research agent that thinks, reads, dreams, and writes — modeled after the cognitive rhythms of a human scientist.</em>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#the-daily-cycle">Daily Cycle</a> •
    <a href="#modules">Modules</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#license">License</a>
  </p>
</p>

---

AutoScientist is an **autonomous scientific research system** that builds and maintains a knowledge graph by reading papers, forming hypotheses, dreaming about connections between ideas, and consolidating knowledge — all on a continuous daily schedule, without human intervention.

It doesn't just retrieve information. It **thinks** about it.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🌐 **Knowledge Graph** | NetworkX-backed graph with typed nodes (concepts, hypotheses, questions, syntheses) and typed edges (supports, causes, contradicts, analogy) |
| 🔍 **Autonomous Research** | Searches Wikipedia & arXiv, reads articles, extracts concepts, and links them into the graph |
| 🌙 **Dream Cycles** | Walks the knowledge graph at night, finding unexpected connections between disparate ideas |
| 🧪 **Computational Sandbox** | Auto-generates and runs Python experiments to test hypotheses |
| 🤔 **Thinker Module** | 5 reasoning patterns: dialectical, analogical, reductive, experimental, integrative |
| ✍️ **Writing Phase** | Forces clarity by synthesizing ideas into structured essays |
| 📓 **Research Notebook** | Persistent journal: morning entries, field notes, evening reflections, breakthroughs |
| 💡 **Delayed Insight Buffer** | Near-miss idea pairs are saved and re-evaluated as new knowledge arrives — mimicking "shower insights" |
| 🔄 **Self-Regulating Knowledge** | Confidence decay on stale nodes forces re-verification; working memory biases active threads |
| 💬 **Conversational Interface** | Chat with the scientist through a web UI |
| ⚡ **FAISS-Backed Embeddings** | Scalable vector similarity search for thousands of concepts |

## Quickstart

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** running locally with at least one model pulled (default: `llama3.1:8b`)

### Installation

```bash
git clone https://github.com/yourusername/autoscientist.git
cd autoscientist
pip install -r requirements.txt
```

### Bootstrap a Research Brain

```bash
# Start with any research question
python bootstrap.py "How does sleep contribute to creative problem-solving?"

# Or fork from a pre-built template brain
python bootstrap.py --template general_scientist "Your question here"
```

### Launch the Web UI

```bash
python gui/app.py
```

Then open **http://localhost:5000** in your browser. You'll see the knowledge graph, notebook, and chat interface.

### Start the Autonomous Scheduler

```bash
python -m scheduler.scheduler
```

The system will now run on a daily cycle automatically.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        AutoScientist                               │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Reader   │→ │ Ingestor │→ │  Brain   │← │ Insight Buffer   │ │
│  │(Wikipedia│  │(extract  │  │(NetworkX │  │(near-miss pairs  │ │
│  │ arXiv)   │  │ nodes &  │  │ graph +  │  │ re-evaluated     │ │
│  └──────────┘  │ edges)   │  │ FAISS)   │  │ each cycle)      │ │
│                └──────────┘  └────┬─────┘  └──────────────────┘ │
│                                   │                              │
│       ┌───────────────────────────┼───────────────────────┐      │
│       │                           │                       │      │
│  ┌────▼─────┐  ┌──────────┐  ┌───▼──────┐  ┌──────────┐ │      │
│  │ Dreamer  │  │ Thinker  │  │Consolida-│  │ Sandbox  │ │      │
│  │(night    │  │(5 reason-│  │tor (merge│  │(run code │ │      │
│  │ walks)   │  │ patterns)│  │ & decay) │  │ tests)   │ │      │
│  └────┬─────┘  └──────────┘  └──────────┘  └──────────┘ │      │
│       │                                                   │      │
│  ┌────▼─────┐  ┌──────────┐  ┌──────────┐               │      │
│  │ Observer │  │ Notebook │  │Researcher│               │      │
│  │(track    │  │(journal) │  │(search & │               │      │
│  │ progress)│  └──────────┘  │ verify)  │               │      │
│  └──────────┘                └──────────┘               │      │
│       └───────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    Scheduler (cron)                       │    │
│  │  09:00 Research → 11:00 Think → 14:00 Read → 16:00 Write│    │
│  │  20:00 Consolidate → 23:00 Dream                        │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │             llm_utils (unified LLM layer)                │    │
│  │  Role-based model selection · Robust JSON parsing        │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## The Daily Cycle

AutoScientist operates on a **circadian rhythm** — six phases that mirror a human researcher's day:

| Time | Phase | What Happens |
|------|-------|-------------|
| **09:00** | 🔬 Research | Picks top questions from the observer agenda, searches the web/arXiv, extracts relevant concepts into the graph |
| **11:00** | 🤔 Thinking | Selects an open question and applies one of 5 reasoning strategies (dialectical, analogical, reductive, experimental, integrative) |
| **14:00** | 📖 Reading | Absorbs articles from the reading list (Wikipedia, arXiv), adds new nodes and cross-domain connections |
| **16:00** | ✍️ Writing | Synthesizes accumulated knowledge into structured essays — forces the system to articulate what it actually knows |
| **20:00** | 🔧 Consolidation | Merges duplicate nodes, creates synthesis/abstraction nodes, detects gaps, decays stale confidence, re-evaluates insight buffer |
| **23:00** | 🌙 Dreaming | Random-walks the knowledge graph, finding unexpected analogies and connections across domains |

## Modules

### Core

| Module | Purpose |
|--------|---------|
| `graph/brain.py` | Knowledge graph — nodes, edges, types, mission tracking, working memory |
| `embedding.py` | Sentence embedding via `sentence-transformers` |
| `embedding_index.py` | FAISS-backed vector index for fast similarity search |
| `llm_utils.py` | Unified LLM interface — role-based model selection, robust JSON parsing |
| `config.py` | Threshold tuning and per-role model configuration |
| `persistence.py` | Atomic JSON writes to prevent corruption |

### Cognitive Modules

| Module | Purpose |
|--------|---------|
| `dreamer/` | Night-time graph walks with 3 modes: focused, wandering, transitional |
| `thinker/` | Structured reasoning — 5 patterns, auto-selects the best strategy per question |
| `consolidator/` | Evening knowledge maintenance — merge duplicates, synthesize, detect gaps, decay confidence |
| `insight_buffer.py` | Delayed insight mechanism — saves near-miss pairs, re-evaluates as graph grows |
| `observer/` | Meta-cognitive monitoring — tracks emergence, coherence, and mission progress |

### Knowledge Acquisition

| Module | Purpose |
|--------|---------|
| `reader/` | Reads Wikipedia articles and arXiv papers, manages a prioritized reading list |
| `researcher/` | Active research — generates search queries, evaluates relevance, extracts findings |
| `ingestion/ingestor.py` | Converts raw text into graph nodes and edges with typed relationships |

### Interface & Scheduling

| Module | Purpose |
|--------|---------|
| `gui/app.py` | Flask + SocketIO web UI — graph visualization, notebook viewer, chat interface |
| `notebook/` | Persistent research journal with typed entries (morning, field notes, evening, breakthrough) |
| `conversation/` | Chat with the scientist — asks questions, ingests relevant responses into the graph |
| `scheduler/` | APScheduler-based daily cycle automation |
| `bootstrap.py` | Dynamic brain initialization — decomposes any research question into domains and seeds the graph |
| `build_template.py` | Creates reusable template brains from general knowledge |

## Configuration

### Model Selection (`config.py`)

AutoScientist uses **role-based model routing** — different cognitive tasks can use different LLM models:

```python
class ModelConfig:
    CREATIVE     = "llama3.1:8b"   # Dreaming, synthesis, analogies
    PRECISE      = "llama3.1:8b"   # JSON extraction, factual answers
    CODE         = "llama3.1:8b"   # Sandbox code generation
    REASONING    = "llama3.1:8b"   # Thinker, chain-of-thought
    CONVERSATION = "llama3.1:8b"   # Chat interface
```

Swap in larger models for critical tasks:
```python
MODELS.CREATIVE  = "llama3.1:70b"   # Better dreaming with bigger model
MODELS.PRECISE   = "qwen2.5:7b"     # Faster JSON extraction
```

### Threshold Tuning (`config.py`)

```python
class ThresholdConfig:
    MERGE_NODE          = 0.80   # Cosine similarity to merge near-duplicate nodes
    DUPLICATE_MERGE     = 0.88   # Strict duplicate detection
    WEAK_EDGE           = 0.60   # Minimum similarity for associative edges
    COHERENCE           = 0.65   # Cross-domain insight quality threshold
    GAP_CONFIDENCE      = 0.75   # Confidence needed to infer gap nodes
```

### Insight Buffer Tuning (`insight_buffer.py`)

```python
BUFFER_LOW       = 0.45   # Minimum similarity to enter the buffer (below WEAK_EDGE)
MAX_EVALUATIONS  = 10     # Prune after this many re-evaluations without improvement
MAX_BUFFER_SIZE  = 200    # Hard cap on buffered pairs
NEIGHBOR_BOOST   = 0.05   # Similarity bonus per shared neighbor
```

## Knowledge Graph Structure

### Node Types

| Type | Description |
|------|-------------|
| `concept` | A factual or theoretical idea extracted from text |
| `hypothesis` | A testable claim with predicted answer and test method |
| `question` | An open question generated by dreaming or research |
| `answer` | A node that resolves a question |
| `synthesis` | Emergent idea created by combining multiple nodes |
| `gap` | Inferred missing link between two connected ideas |
| `mission` | The central research question |
| `empirical` | Result from a computational sandbox test |

### Edge Types

| Type | Description |
|------|-------------|
| `supports` | Evidence or reasoning that backs another idea |
| `causes` | Causal relationship |
| `contradicts` | Logical tension or opposing evidence |
| `surface_analogy` | Shared vocabulary or theme across domains |
| `structural_analogy` | Same relational pattern (A:B :: X:Y) |
| `deep_isomorphism` | Formal mathematical or logical equivalence |
| `associated` | Weak associative link based on embedding similarity |

## Project Structure

```
autoscientist/
├── bootstrap.py           # Initialize a research brain from any question
├── build_template.py      # Build reusable template brains
├── config.py              # Thresholds + model configuration
├── embedding.py           # Sentence embedding interface
├── embedding_index.py     # FAISS vector index
├── insight_buffer.py      # Delayed insight mechanism
├── llm_utils.py           # Unified LLM layer (llm_call, require_json)
├── persistence.py         # Atomic JSON persistence
│
├── graph/
│   └── brain.py           # Core knowledge graph (NetworkX)
│
├── consolidator/
│   └── consolidator.py    # Evening consolidation (7 steps)
│
├── conversation/
│   └── conversation.py    # Chat interface with graph integration
│
├── dreamer/
│   └── dreamer.py         # Night-time graph walks
│
├── gui/
│   ├── app.py             # Flask web application
│   └── templates/
│       └── index.html     # Web UI
│
├── ingestion/
│   └── ingestor.py        # Text → graph node/edge extraction
│
├── notebook/
│   └── notebook.py        # Research journal
│
├── observer/
│   └── observer.py        # Meta-cognitive monitoring
│
├── reader/
│   └── reader.py          # Wikipedia/arXiv reader
│
├── researcher/
│   └── researcher.py      # Active research agent
│
├── sandbox/
│   └── sandbox.py         # Computational hypothesis testing
│
├── scheduler/
│   └── scheduler.py       # Daily cycle automation
│
├── tests/
│   └── test_embedding_index.py
│
├── thinker/
│   └── thinker.py         # Structured reasoning (5 patterns)
│
├── requirements.txt
└── LICENSE                # GPL-3.0
```

## Self-Regulating Knowledge

AutoScientist treats knowledge as a **decaying asset**, not a permanent record:

- **Confidence Decay**: Nodes not re-verified within 3 days lose `source_quality` at 2% per day. Dream-synthesized nodes (low initial quality) decay 2× faster. Floor at 0.05.
- **Edge Pruning**: Weak edges with confidence below threshold are removed during consolidation.
- **Working Memory Bias**: The dreamer has a 30% probability of starting its walk from a node currently in working memory, keeping active research threads alive.
- **Delayed Insights**: Near-miss pairs (similarity 0.45-0.59) are buffered and re-evaluated each cycle. Shared neighbors boost the score. After 10 failed re-evaluations, pairs are pruned.

## Prompt Engineering

All 48 LLM prompts have been calibrated with:

- **Scored rubrics** — every numeric output (strength, confidence, coherence) has a labeled scale with concrete examples at each level
- **Negative examples** — synthesis, gap, and abstraction prompts include "bad example" demonstrations to prevent summarization
- **Grading definitions** — categorical outputs (none/partial/strong) are defined with explicit tests ("could you write a conclusion?")
- **Cross-domain depth verification** — the observer cross-checks claimed analogy depth against actual content

## Running Tests

```bash
pytest tests/ -v
```

## Data Persistence

All state is persisted to `data/`:

```
data/
├── brain.json              # Full knowledge graph
├── observer.json           # Observer state (agenda, signals)
├── embedding_index/        # FAISS index files
├── insight_buffer.json     # Pending near-miss pairs
├── consolidation_latest.json
└── daily_new_nodes.json    # Ledger for daily tracking
```

Logs are written to `logs/`:
```
logs/
├── cycle_log.json          # Scheduler phase log
├── research_log.json       # Research session details
├── notebook.json           # All journal entries
└── sandbox_results.json    # Computational experiment results
```

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>"The mind, once stretched by a new idea, never returns to its original dimensions."</em><br/>
  — Oliver Wendell Holmes Sr.
</p>
