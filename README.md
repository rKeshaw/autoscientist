# AutoScientist: An Autonomous Agent for Long-Horizon Scientific Discovery via Connectome Memory

AutoScientist is an autonomous research system designed to accelerate scientific discovery over long horizons. Rather than relying on static prompting, append-only context windows, or flat vector search—which suffer from catastrophic forgetting and context drift—AutoScientist structures its evolving knowledge into a biologically inspired **graph connectome memory**.

The system operates on continuous diurnal research cycles: reading literature, deriving hypotheses, evaluating claims through an adversarial System 2 critic, executing computational experiments in an execution sandbox, and running multi-pass sleep consolidation to maintain structural coherence and resolve contradictions.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Component Overview](#component-overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Running Autonomous Research Cycles](#running-autonomous-research-cycles)
- [Reproducing Benchmarks and Results](#reproducing-benchmarks-and-results)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Overview

Modern scientific research faces a dual bottleneck: literature volume is growing faster than human synthesis capacity, while standard language model agents degrade over extended reasoning horizons. Standard retrieval-augmented generation (RAG) retrieves isolated text chunks without understanding structural, causal, or contradictory dependencies, leading to reasoning loops and forgotten insights.

AutoScientist addresses this by modeling memory as a dynamic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$:

- **Propositions as Nodes:** Concepts, hypotheses, empirical results, research questions, synthesis hubs, and epistemic gaps.
- **Epistemic Relations as Edges:** Directed relationships specifying support, contradiction, causation, structural analogy, and deep isomorphism.
- **Neuromodulated Working Memory:** Bounded active memory capacity ($W_{\max}$) regulated by synthetic dopamine (discovery reward) and frustration (experiment failure) signals.
- **Diurnal Consolidation:** Nightly sleep passes (inspired by slow-wave NREM and REM sleep) that merge near-duplicates, prune decayed synapses, resolve contradictions, and promote incubated insights.

---

## Key Features

- **Connectome Memory Graph:** Dynamic network of typed scientific claims and explicit epistemic dependencies, preventing knowledge collapse over dozens of discovery cycles.
- **Predictive Ingestion:** Active-inference reading module that generates top-down expectations prior to reading, extracting propositions and measuring prediction surprise.
- **Dual-Process Gating (System 2 Critic):** High-stakes claims (hypotheses, cross-domain analogies) must pass adversarial multi-turn review before entering permanent memory.
- **Computational Sandbox:** Auto-generates executable Python simulations (numerical models, differential equations, statistical tests) to empirically verify or falsify hypotheses.
- **Contextual Bandit Reasoning:** Reinforcement learning policy (`CognitivePolicy`) that dynamically selects reasoning patterns (reductive, analogical, dialectical, experimental, integrative) based on graph context and dopamine reward.
- **6-Pass Sleep Consolidation:** Nocturnal graph refinement algorithm executing deduplication, abstraction synthesis, gap detection, contradiction resolution, synaptic decay, and insight buffer promotion.
- **Salience Scheduler:** Priority-queue event loop that dynamically interrupts routine background tasks when critical breakthroughs or anomalies occur.

---

## System Architecture

```
                       ┌────────────────────────────────────────┐
                       │          Salience Scheduler            │
                       │       (Priority Event Loop)            │
                       └──────────────────┬─────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
   ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
   │ Predictive      │           │ Thinker         │           │ Dreamer         │
   │ Reader          │           │ (Procedural RL) │           │ (Hippocampal    │
   │ & Ingestor      │           └────────┬────────┘           │  Replay)        │
   └────────┬────────┘                    │                    └────────┬────────┘
            │                             │                             │
            │              ┌──────────────┴──────────────┐              │
            │              ▼                             ▼              │
            │     ┌─────────────────┐           ┌─────────────────┐     │
            │     │ System 2 Critic │           │ Computational   │     │
            │     │ (Adversarial)   │           │ Sandbox         │     │
            │     └────────┬────────┘           └────────┬────────┘     │
            │              │                             │              │
            └──────────────┼─────────────────────────────┼──────────────┘
                           │                             │
                           ▼                             ▼
            ┌───────────────────────────────────────────────────────────┐
            │                     Connectome Memory                     │
            │              (Nodes, Edges, Synaptic Weights)             │
            └─────────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────┐
            │                  6-Pass Sleep Consolidator                │
            │ (Deduplication • Abstraction • Gaps • Contradiction •     │
            │  Synaptic Decay • InsightBuffer Promotion)                │
            └───────────────────────────────────────────────────────────┘
```

---

## Component Overview

### 1. Connectome Memory (`graph/brain.py`)
Maintains the directed graph structure with vector indexing via FAISS (`embedding_index.py`).
- **Node Types:** `concept`, `hypothesis`, `empirical`, `synthesis`, `gap`, `question`, `mission`.
- **Edge Types:** `supports`, `contradicts`, `causes`, `structural_analogy`, `deep_isomorphism`, `associated`, `toward_mission`, `empirically_tested`.
- Tracks neuromodulator states (dopamine $\mathcal{D} \in [0, 1]$, frustration $\mathcal{F} \in [0, 1]$), edge traversal frequencies, and synaptic decay rates.

### 2. Predictive Reader & Ingestor (`reader/`, `ingestion/`)
- Autonomous search across arXiv, Wikipedia, and targeted web sources.
- Implements active inference: the reader predicts what a document should argue before absorption, computes expectation surprise, and links new propositions into existing clusters.

### 3. Thinker & Cognitive Policy (`thinker/`)
- Generates new conjectures and subquestions conditioned on current graph frontiers and unresolved gaps.
- Uses a contextual bandit (`thinker/policy.py`) over five cognitive patterns:
  - `reductive`: Break down complex claims into fundamental physical/mathematical constituents.
  - `analogical`: Transfer structural mechanisms across distinct scientific domains.
  - `dialectical`: Surface underlying assumptions and resolve thesis-antithesis pairs.
  - `experimental`: Formulate concrete, testable computational protocols.
  - `integrative`: Synthesize disconnected clusters into higher-order theoretical principles.

### 4. System 2 Critic (`critic/`)
- Intercepts generated hypotheses and analogies before graph insertion.
- Evaluates logical consistency, empirical plausibility, and novelty through adversarial challenge-defense dialogue.
- Issues verdicts: `ACCEPT`, `REFINE` (up to 2 revision loops), `REJECT`, or `DEFER` (routed to `InsightBuffer`).

### 5. Computational Sandbox (`sandbox/`)
- Converts experimental hypotheses into executable Python code with automated dependency handling.
- Executes simulations within bounded runtime and memory limits.
- Evaluates outputs against test criteria, returning typed outcomes (`supports`, `contradicts`, `inconclusive`, `error`) and generating visual plots.

### 6. Sleep Consolidator (`consolidator/`)
Executes nightly 6-pass consolidation on the connectome:
1. **Deduplication:** Merges semantically redundant nodes ($\text{cosine similarity} \ge 0.88$).
2. **Abstraction Synthesis:** Identifies dense topological cliques and constructs abstract synthesis nodes.
3. **Gap Inference:** Detects missing bridge concepts between co-activated clusters.
4. **Contradiction Resolution:** Detects and flags conflicting empirical claims for dialectical arbitration.
5. **Synaptic Weight Decay:** Exponentially attenuates unreinforced edges and prunes inactive connections.
6. **InsightBuffer Promotion:** Evaluates incubated near-miss hypothesis pairs against newly acquired evidence.

### 7. Observer & Metacognition (`observer/`)
- Monitors graph health, community modularity ($Q$), clustering coefficients ($C$), and degree distributions.
- Detects emergence events: mission advances, persistent contradictions, recurring questions, and long-incubation resolutions.

---

## Installation

### Prerequisites

- **Linux** (tested on Ubuntu 20.04/22.04) or **macOS**
- **Python 3.10+**
- **[Ollama](https://ollama.ai)** running locally or accessible via network

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/autoscientist.git
cd autoscientist

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Pull Required Local Models

AutoScientist defaults to Qwen 2.5 models for reasoning and coding. Pull the models using Ollama:

```bash
# General reasoning, extraction, dreaming, and critic
ollama pull qwen2.5:32b

# Code generation for the computational sandbox
ollama pull qwen2.5-coder:32b
```

*Note: For lower VRAM environments (e.g., 16GB–24GB), you can switch to `14b` or `7b` models in `config.py`:*
```bash
ollama pull qwen2.5:14b
ollama pull qwen2.5-coder:14b
```

---

## Quickstart

### 1. Bootstrap a New Research Mission

Initialize a connectome around a central scientific research question. The bootstrap routine decomposes the question into domains, pulls foundational papers, extracts core concepts, and performs an initial dream and consolidation pass:

```bash
python bootstrap.py "How do critical phase transitions in physical systems relate to edge-of-chaos dynamics and computational capacity in neural networks?"
```

This creates the persistent graph state in `data/brain.json`, vector index in `data/embedding_index`, and initial agenda in `data/observer.json`.

### 2. Launch the Interactive Web GUI

Inspect the real-time connectome graph, reading queue, running hypothesis, and experimental logs:

```bash
python gui/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Running Autonomous Research Cycles

### Multi-Loop Autonomous Execution

Run continuous diurnal cycles sequentially (each cycle executes dream $\rightarrow$ research $\rightarrow$ thinking $\rightarrow$ reading $\rightarrow$ writing $\rightarrow$ consolidation):

```bash
# Run 5 complete research cycles
python scheduler/scheduler.py --mode cycle --loops 5
```

Alternatively, use the provided pipeline shell script:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Running Individual Cognitive Phases Manually

You can trigger specific phases directly for testing or inspection:

```bash
# Run hypothesis generation and System 2 review
python scheduler/scheduler.py --mode thinking

# Run REM/NREM random graph walk and analogy discovery
python scheduler/scheduler.py --mode dream

# Run 6-pass sleep consolidation
python scheduler/scheduler.py --mode consolidation

# Inspect current connectome and agenda status
python scheduler/scheduler.py --mode status
```

---

## Reproducing Benchmarks and Results

The repository includes standalone benchmark scripts to evaluate each architectural component and reproduce experimental findings:

### 1. Connectome Topology Evolution (20-Cycle Horizon)
Measures small-world coefficient ($S$), modularity ($Q$), characteristic path length ($L$), and clustering coefficient ($C$) over extended runs:

```bash
python benchmark/analyze_brain_topology_expanded.py
```
*Key finding: Over 20 cycles, small-world coefficient grows from $S=2.90 \to 34.02$, path length shortens from $5.10 \to 4.08$, and modularity stabilizes at $Q=0.827$, demonstrating scale-free, small-world connectome organization without catastrophic forgetting.*

### 2. Sleep Consolidation Ablation Suite
Ablates individual passes of the nocturnal consolidation pipeline (zero consolidation, no decay, no deduplication, no gap detection):

```bash
python benchmark/analyze_sleep_consolidation_full_scope.py
```
*Key finding: Disabling consolidation degrades modularity by 26.0% ($Q=0.612$) and completely blinds the agent to epistemic contradictions, while full 6-pass consolidation actively maintains topological health.*

### 3. Representation Baseline Comparison (Graph vs. RAG vs. Flat-File)
Evaluates retrieval continuity, multi-hop reasoning, contradiction detection, and decoy resistance across memory architectures:

```bash
python benchmark/test_phase3_representation_deep_dive.py
```
*Key finding: The connectome graph achieves 100% multi-hop path continuity and 100% contradiction recall, whereas standard vector RAG suffers from 0% contradiction recall and 29.2% decoy contamination.*

### 4. Neuromodulatory Working Memory & Capacity Sweep
Sweeps working memory capacities ($W_{\max} \in [3, 100]$) and evaluates dopamine-modulated learning rates:

```bash
python benchmark/test_track5_neuromodulatory_working_memory.py
```
*Key finding: An active working memory capacity of $W_{\max}=7\text{--}10$ eliminates thrashing while maintaining minimal token overhead, and dopamine-based reward accelerates policy convergence by 19.7%.*

### 5. Automated Unit Test Suite
Runs the full functional test suite verifying brain modes, FAISS vector index, insight buffer, LLM utilities, and sandbox execution:

```bash
pytest tests/ -v
```

---

## Configuration

All system thresholds, model assignments, and critic settings are managed in `config.py`:

```python
# Model Routing (Ollama tag names)
class ModelConfig:
    CREATIVE     = "qwen2.5:32b"       # Dreamer analogies, synthesis
    PRECISE      = "qwen2.5:32b"       # Information extraction, JSON formatting
    CODE         = "qwen2.5-coder:32b" # Sandbox simulation scripts
    REASONING    = "qwen2.5:32b"       # Thinker deliberate reasoning
    CRITIC       = "qwen2.5:32b"       # System 2 adversarial evaluation

# Epistemic & Topological Thresholds (Cosine similarity)
class ThresholdConfig:
    MERGE_NODE      = 0.72   # Near-duplicate merge threshold
    DUPLICATE_MERGE = 0.88   # Exact duplicate merge threshold
    WEAK_EDGE       = 0.58   # Minimum similarity for associative links
    COHERENCE       = 0.65   # Cross-domain insight threshold
    GAP_CONFIDENCE  = 0.75   # Minimum confidence to infer structural gaps
    CONTRADICTION   = 0.60   # Contradiction detection threshold

# System 2 Dual-Process Gating
class CriticConfig:
    ACTIVATION_THRESHOLD   = 0.65   # Minimum importance to trigger System 2
    MAX_DIALOGUE_TURNS     = 3      # Maximum adversarial debate rounds
    ACCEPT_CONFIDENCE_FLOOR = 0.50  # Minimum confidence to accept a claim
```

---

## Repository Structure

```
autoscientist/
├── bootstrap.py              # Dynamic mission bootstrapping from literature
├── config.py                 # System thresholds, model routing, critic settings
├── run_pipeline.sh           # Shell script for automated multi-cycle execution
├── requirements.txt          # Python dependencies
│
├── graph/
│   ├── brain.py              # Connectome graph core, neuromodulators, working memory
│   └── episodic.py           # Chronological episodic event logging
│
├── thinker/
│   ├── thinker.py            # Hypothesis generation and reasoning patterns
│   └── policy.py             # Contextual bandit RL policy for pattern selection
│
├── critic/
│   └── critic.py             # Dual-process System 2 adversarial evaluation
│
├── sandbox/
│   └── sandbox.py            # Isolated Python simulation execution environment
│
├── consolidator/
│   └── consolidator.py       # 6-pass nocturnal sleep consolidation engine
│
├── dreamer/
│   └── dreamer.py            # REM/NREM random graph walks and cross-domain synthesis
│
├── reader/
│   └── reader.py             # Literature retrieval (arXiv, Wikipedia) and queue management
│
├── ingestion/
│   └── ingestor.py           # Predictive processing extraction and graph insertion
│
├── observer/
│   └── observer.py           # Epistemic monitoring, emergence detection, agenda tracking
│
├── notebook/
│   └── notebook.py           # Running hypothesis synthesis and research journal
│
├── scheduler/
│   └── scheduler.py          # Salience network priority-queue event loop
│
├── gui/
│   ├── app.py                # Flask web interface for connectome visualization
│   └── templates/            # HTML/JS templates for interactive graph rendering
│
├── benchmark/                # Quantitative benchmark and ablation suite
│   ├── analyze_brain_topology_expanded.py
│   ├── analyze_sleep_consolidation_full_scope.py
│   ├── test_phase3_representation_deep_dive.py
│   └── test_track5_neuromodulatory_working_memory.py
│
├── tests/                    # Unit and integration test suite
│   ├── test_brain_modes.py
│   ├── test_embedding_index.py
│   ├── test_insight_buffer.py
│   ├── test_llm_utils.py
│   └── test_thinker_sandbox.py
│
├── data/                     # Persistent storage (created at runtime)
│   ├── brain.json            # Knowledge graph state
│   ├── observer.json         # Observer agenda and emergence feed
│   └── embedding_index/      # FAISS vector index files
│
└── logs/                     # Execution and cycle logs (created at runtime)
    ├── cycle_log.json        # Per-cycle telemetry
    ├── sandbox_log.json      # Code execution results and verdicts
    └── notebook.json         # Versioned running hypothesis log
```

---

## License

This project is licensed under the terms of the GNU General Public License v3.0 ([LICENSE](LICENSE)).
