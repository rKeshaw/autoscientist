# Dimension 1 — Knowledge Graph Quality Benchmark

## What This Tests

Five targeted tests that collectively answer: **Is the knowledge graph
actually good?** The graph is the foundation. If nodes are vague, edges
wrong, clusters meaningless, or duplicates rampant, every other module
fails on bad inputs.

| Test | File | What It Measures | Pass Criterion |
|------|------|-----------------|----------------|
| T1: Node Quality | `test_d1_node_quality.py` | Are extracted nodes rich conceptual statements or keyword noise? | ≥ 80% nodes score 4+/5 |
| T2: Duplicate Rate | `test_d1_duplicate_rate.py` | Does the dedup pipeline actually merge near-duplicates? | < 5% duplicate pairs post-consolidation |
| T3: Edge Accuracy | `test_d1_edge_accuracy.py` | Are relationship types (supports/causes/contradicts/analogy) correctly extracted? | ≥ 70% type accuracy; no type with recall < 50% |
| T4: Cluster Coherence | `test_d1_cluster_coherence.py` | Are domain cluster assignments semantically meaningful? | Intra-sim ≥ 0.55; inter-sim ≤ 0.40; assignment acc ≥ 75% |
| T5: Contradiction Detection | `test_d1_contradiction_detection.py` | Does the system correctly find contradictions and ignore compatible pairs? | Precision ≥ 80%; Recall ≥ 70% |

## Setup

All tests run FROM the AutoScientist project root. They import directly
from the project source — no separate package installation needed.

```bash
# From project root
pip install -r requirements.txt   # if not already done
ollama pull llama3.1:70b          # or your preferred 70b judge model
```

## Running

### Run all five tests at once (recommended)

```bash
# From project root
bash benchmark/dim1/run_d1_all.sh --judge-model llama3.1:70b
```

This runs all five tests sequentially and produces an aggregate report.
**Expected runtime: 45–90 minutes** depending on your hardware and model speed.

Results are written to `benchmark/dim1/results/`.

### Run individual tests

```bash
# Test 1: Node Quality
python benchmark/dim1/test_d1_node_quality.py \
    --judge-model llama3.1:70b \
    --out benchmark/dim1/results/d1_node_quality.json

# Test 2: Duplicate Rate
python benchmark/dim1/test_d1_duplicate_rate.py \
    --judge-model llama3.1:70b \
    --out benchmark/dim1/results/d1_duplicate_rate.json

# Test 3: Edge Accuracy
python benchmark/dim1/test_d1_edge_accuracy.py \
    --judge-model llama3.1:70b \
    --out benchmark/dim1/results/d1_edge_accuracy.json

# Test 4: Cluster Coherence
python benchmark/dim1/test_d1_cluster_coherence.py \
    --judge-model llama3.1:70b \
    --out benchmark/dim1/results/d1_cluster_coherence.json

# Test 5: Contradiction Detection
python benchmark/dim1/test_d1_contradiction_detection.py \
    --judge-model llama3.1:70b \
    --out benchmark/dim1/results/d1_contradiction_detection.json

# Aggregate report (after all 5 tests)
python benchmark/dim1/report_d1.py \
    --results-dir benchmark/dim1/results/ \
    --out benchmark/dim1/results/d1_report.json
```

### Speed optimizations

**Re-use cached nodes** (avoids re-fetching Wikipedia + re-ingesting):

```bash
# First run builds the cache
python benchmark/dim1/test_d1_node_quality.py \
    --judge-model llama3.1:70b \
    --cache benchmark/dim1/results/d1_node_cache.json

# Subsequent runs load from cache
python benchmark/dim1/test_d1_node_quality.py \
    --judge-model llama3.1:70b \
    --skip-ingest \
    --cache benchmark/dim1/results/d1_node_cache.json
```

**Reduce judged nodes** (faster, less accurate):

```bash
python benchmark/dim1/test_d1_node_quality.py \
    --judge-model llama3.1:70b \
    --max-nodes-per-article 20   # default is 50
```

## Output Structure

Each test writes a JSON report with:
- `summary` — key metrics and PASS/FAIL verdict
- `config` — parameters used
- `per_X` — per-article, per-type, or per-cluster breakdowns
- `raw_judgments` — every LLM judge call with full reasoning
- `worst_nodes` / `mislabeled_nodes` / `missed_contradictions` — failure analysis

The aggregator (`report_d1.py`) reads all five and writes `d1_report.json`
with an overall Dimension 1 verdict.

## Benchmark Corpus

Five Wikipedia articles chosen to cover distinct domains with known
conceptual relationships:

| Article | Domain | Key Concepts Tracked |
|---------|---------|---------------------|
| DNA | biology | double helix, base pairing, replication, transcription, genetic code |
| Thermodynamics | physics | entropy, laws, heat, work, free energy |
| Natural Selection | evolutionary_biology | variation, fitness, adaptation, selection pressure |
| Artificial Neural Network | computer_science | weights, backprop, gradient descent, overfitting |
| Game Theory | economics | Nash equilibrium, payoff matrix, prisoner's dilemma |

## Judge Model Notes

Tests use an LLM-as-judge pipeline. Recommended models:

| Model | Notes |
|-------|-------|
| `llama3.1:70b` | Recommended — good calibration on rubric-based scoring |
| `llama3.3:70b` | Also good |
| `qwen2.5:72b` | Strong on structured JSON output |
| `llama3.1:8b` | Fast but less calibrated; use only for development runs |

The judge prompt includes explicit rubrics with labeled scales and
concrete examples at each level to minimize calibration variance.

## Interpreting Results

**Node Quality (T1)**
- Mean score < 3.0: Extraction is generating keyword noise. Check `worst_nodes`.
- Mean score 3.0–4.0: Functional but nodes lack depth. Review `improved_version` fields.
- Mean score > 4.0: Nodes are genuinely conceptual.

**Duplicate Rate (T2)**
- `new_nodes_on_reingestion` is the most diagnostic metric. If re-ingesting
  the same article creates many new nodes, the dedup threshold (0.80) may
  be too loose or the enrichment path has a bug.

**Edge Accuracy (T3)**
- Confusion matrix shows systematic errors. Common failures:
  - `contradicts` predicted as `associated` → contradiction threshold too high
  - `analogy:isomorphism` predicted as `analogy:structural` → depth detection weak
  - `causes` predicted as `supports` → causal direction detection weak

**Cluster Coherence (T4)**
- Low intra-cluster similarity (< 0.40) means the LLM cluster labels are
  inconsistent — same domain gets multiple names (`ML`, `machine_learning`, etc.)
- High inter-cluster similarity (> 0.50) means domain separation is poor —
  all nodes land in similar embedding space regardless of label.

**Contradiction Detection (T5)**
- Low recall (many missed): Contradiction similarity threshold (0.65) is too
  high, or the LLM confirmation step is too conservative.
- Low precision (many false alarms): Compatible pairs with similar vocabulary
  are being incorrectly flagged.
