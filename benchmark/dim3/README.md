# Dimension 3 - Thinker / Structured Reasoning Benchmark

## Revised Position In The Roadmap

Dimension 3 covers deliberate reasoning quality for `Thinker`.

In the current implementation, D3 answers:
**Does the structured reasoning module choose sensible reasoning strategies,
decompose hard questions well, stay coherent across rounds, and produce
mission-useful outputs?**

## Current Implemented Scope

The Dimension 3 suite includes six tests:

| Test | File | What It Measures |
|------|------|------------------|
| T1 | `test_d3_pattern_selection.py` | Whether Thinker picks a plausible reasoning strategy |
| T2 | `test_d3_reductive_sufficiency.py` | Whether reductive decomposition produces answer-bearing sub-questions |
| T3 | `test_d3_actionability.py` | Whether thinker outputs suggest concrete next steps |
| T4 | `test_d3_cross_round_coherence.py` | Whether multi-round thinking stays on-topic and cumulative |
| T5 | `test_d3_specificity_lift.py` | Whether later rounds become more mission-specific than the first |
| T6 | `test_d3_subquestion_utility.py` | Whether generated sub-questions are useful for later research |

## Running

Run the full Dimension 3 suite:

```bash
bash benchmark/dim3/run_d3_all.sh --judge-model llama3.1:70b
```

Prepare a Dim 3 shared graph ahead of time:

```bash
python benchmark/dim3/prep_d3_graph.py
```

Aggregate results after the suite runs:

```bash
python benchmark/dim3/report_d3.py
```

## Notes

The suite isolates the procedural policy file per test so benchmark runs do not
mutate the live `data/policy.json` used by normal operation.

When available, the suite reuses the shared graph prepared for Dimension 2 to
avoid rebuilding the same background knowledge repeatedly.
