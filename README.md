
‎README.md‎
+111
Lines changed: 111 additions & 0 deletions
Original file line number	Diff line number	Diff line change
# AutoScientist

> An autonomous scientific reasoning agent — powered by local LLMs via [Ollama](https://ollama.ai).
AutoScientist runs a mission-driven research loop on your local machine. It reads papers, forms and tests hypotheses, searches the web and arXiv for evidence, runs virtual experiments, and keeps a structured research notebook — all without human intervention.

Every piece of knowledge carries an explicit epistemic label: `grounded` (externally sourced), `prior` (model knowledge, provisional), `speculative`, `contradicted`, or `open`. Prior knowledge guides search and ideation but is never promoted to settled fact without evidence.

---

## How It Works

Each autonomous cycle follows the scientific method:

**Read → Hypothesise → Test → Research → Experiment → Dream → Consolidate → Reflect**

The system is built around a set of collaborating cognitive modules:

| Module | Role |
|--------|------|
| **Brain** | Evidence-grounded knowledge graph (NetworkX) |
| **Ingestor** | LLM-powered text → graph pipeline |
| **Reader** | Absorbs URLs, arXiv papers, and PDFs |
| **Researcher** | Web + arXiv search to ground hypotheses |
| **Thinker** | Structured deliberate reasoning (dialectical, analogical, reductive…) |
| **Dreamer** | Associative ideation via random walks on the graph |
| **Critic** | Adversarial System 2 gating — ACCEPT / REFINE / REJECT / DEFER |
| **Experimenter** | Generates and runs sandboxed Python experiments |
| **Observer** | Research agenda, emergence signals, weekly pivots |
| **Notebook** | Structured research journal |
| **Conversation** | Interactive chat with the scientist persona |

---

## Getting Started

**Requirements:** Python ≥ 3.11, [Ollama](https://ollama.ai) running locally, internet access.

```bash
pip install -r requirements.txt
```

**Verify the installation** (no LLM needed):

```bash
python smoke_workbench.py --mode structural
```

**Start the service:**

```bash
python service_shell.py --port 8000
```

**Set a mission and run:**

```bash
# Set a research mission
curl -X POST http://localhost:8000/mission \
  -H "Content-Type: application/json" \
  -d '{"question": "What mechanisms underlie memory consolidation during sleep?"}'

# Run an autonomous cycle
curl -X POST http://localhost:8000/cycles/run -d '{"count": 1}'

# Chat with the scientist
curl -X POST http://localhost:8000/conversation \
  -d '{"message": "What is the current working hypothesis?"}'
```

To run cycles automatically, pass `--scheduler-interval 60` (seconds per cycle).

---

## Reading List

Drop URLs, arXiv IDs, or PDF paths into `data/reading_list.json` and they will be absorbed on the next cycle:

```json
[
  {"url": "https://arxiv.org/abs/2301.07041"},
  {"url": "https://example.org/paper.pdf"}
]
```

---

## Configuration

Edit `config.py` to change:

- **Models** — assign different Ollama models per role (creative, precise, critic, …)
- **Thresholds** — similarity cutoffs for merging, deduplication, and contradiction detection
- **Critic gating** — which claim types trigger adversarial review

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET/POST` | `/mission` | Get or set the research mission |
| `POST` | `/cycles/run` | Run N autonomous cycles |
| `GET` | `/notebook` | Recent journal entries |
| `POST` | `/conversation` | Chat with the scientist |
| `POST/POST` | `/scheduler/start` `/scheduler/stop` | Autonomous scheduling |

---

> Active development — see [`PLAN.md`](PLAN.md) for the current roadmap.
