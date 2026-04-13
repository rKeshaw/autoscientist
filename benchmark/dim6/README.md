# Dimension 6 - Consolidation & Insight Buffer Benchmark

## Revised Position In The Roadmap

Dimension 6 now covers both sides of offline cognition:
- generative enrichment during consolidation
- maintenance of weak, conflicting, and incubating structure over time

In the current implementation, D6 answers:
**Does the evening consolidation pass create meaningful higher-order structure,
preserve useful tensions, decay stale weak structure, and promote delayed
insights only when later context makes them stronger?**

This suite now reuses the inherited shared benchmark graph from
`benchmark/dim4/shared/`, following the same inheritance pattern used by later
dimensions rather than constructing each D6 test from a tiny standalone brain.

## Current Implemented Scope

The Dimension 6 suite includes seven tests:

| Test | File | What It Measures |
|------|------|------------------|
| T1 | `test_d6_synthesis_genuineness.py` | Whether synthesis outputs are genuinely emergent rather than summaries |
| T2 | `test_d6_abstraction_quality.py` | Whether abstraction outputs capture a real meta-pattern rather than a cheap label |
| T3 | `test_d6_gap_inference.py` | Whether inferred gaps are actual missing mediating links |
| T4 | `test_d6_contradiction_maintenance.py` | Whether unresolved contradictions are kept salient during consolidation |
| T5 | `test_d6_decay_calibration.py` | Whether weak edges decay away while contradiction edges and fresh nodes survive |
| T6 | `test_d6_delayed_insight_promotion.py` | Whether the insight buffer promotes near-miss pairs once graph context strengthens them |
| T7 | `test_d6_time_to_promotion.py` | Whether stagnant pending pairs are pruned after repeated failed evaluations |

## Why The Scope Is Extended

The original D6 plan focused mostly on generative outputs plus decay. That was
not quite enough for the actual `Consolidator` implementation. Consolidation is
also responsible for tension maintenance via `_contradiction_update()`, so the
suite now explicitly measures whether unresolved contradictions are surfaced
instead of quietly fading into the background.

Each metric is also exercised with more than one case type:
- semantic generation tests use graph-derived positive suites and hard
  negatives assembled from real nodes in the inherited brain
- maintenance tests run on top of the inherited graph and add only the minimal
  temporary probes needed for threshold / pruning checks

Out of scope for D6 on purpose:
- near-duplicate merge quality stays in D1
- mission-link recalibration after consolidation is treated as part of D7
  mission tracking rather than core D6 idea-quality evaluation

## Running

Run the full Dimension 6 suite:

```bash
bash benchmark/dim6/run_d6_all.sh --model gemma4:latest --judge-model llama3.1:70b
```

If you omit `--judge-model`, the judge defaults to the same model as
`--model`.

Prerequisite:
`benchmark/dim4/shared/brain.json` and the matching embedding index must
already exist. D6 inherits that shared graph the same way D5 does.

Run an individual semantic test:

```bash
python benchmark/dim6/test_d6_synthesis_genuineness.py \
  --model gemma4:latest \
  --judge-model llama3.1:70b
```

Run an individual maintenance test:

```bash
python benchmark/dim6/test_d6_decay_calibration.py \
  --out benchmark/dim6/results/d6_decay_calibration.json
```

Aggregate results after the suite runs:

```bash
python benchmark/dim6/report_d6.py
```

## Notes

- T1-T3 use the real `Consolidator` LLM path on graph-derived cases resolved
  from the inherited shared brain.
- T4-T7 also start from the inherited graph; the buffer tests remain
  deterministic in their promotion rule so they benchmark graph maintenance
  rather than local LLM variability.
- T5 isolates the consolidation timestamp file inside a temporary directory so
  the benchmark does not perturb normal runtime state.
