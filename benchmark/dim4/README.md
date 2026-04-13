# Dimension 4 - Critic / System 2 Quality Benchmark

## Revised Position In The Roadmap

Dimension 4 covers adversarial review quality for `Critic` (System 2).

In the current implementation, D4 answers:
**Does the Critic improve the system's claims through calibrated gating,
accurate novelty detection, and useful refinement — without just adding
latency and friction?**

## Current Implemented Scope

The Dimension 4 suite includes six tests:

| Test | File | What It Measures |
|------|------|------------------|
| T1 | `test_d4_activation_calibration.py` | Whether the laziness gate correctly routes high vs low-stakes claims |
| T2 | `test_d4_verdict_accuracy.py` | Whether the Critic issues reasonable accept/refine/reject/defer verdicts |
| T3 | `test_d4_novelty_check.py` | Whether novelty detection correctly identifies novel vs redundant claims |
| T4 | `test_d4_refinement_lift.py` | Whether the refinement loop improves weak claims |
| T5 | `test_d4_defer_quality.py` | Whether DEFER verdicts are appropriately uncertain |
| T6 | `test_d4_bypass_safety.py` | Whether bypassed claims are genuinely safe to bypass |

## Running

Run the full Dimension 4 suite:

```bash
bash benchmark/dim4/run_d4_all.sh --judge-model gemma4:latest
```

Prepare a Dim 4 shared graph ahead of time:

```bash
python benchmark/dim4/prep_d4_graph.py
```

Aggregate results after the suite runs:

```bash
python benchmark/dim4/report_d4.py
```

## Notes

- Tests T1 and T6 are **deterministic** (no LLM calls) — they test the
  `needs_review()` logic directly against constructed candidates.
- Tests T2-T5 use the **full LLM pipeline** and may have stochastic variance.
- The dim4 graph is **enriched** beyond dim2/dim3 with four additional domains:
  Reinforcement Learning, Neuroscience, Information Theory, and Complex Systems.
- Verdict accuracy uses **lenient matching** — each test case specifies a set of
  acceptable verdicts rather than requiring an exact match.
