# AutoScientist Benchmark Roadmap (Revised)

This roadmap supersedes the earlier benchmark outline that was written before
`Thinker`, `Critic`, and the newer shared-state integrations became core parts
of the runtime.

The original benchmark plan is still directionally correct, especially for:
- graph quality
- dream quality
- research/reading quality
- consolidation quality
- end-to-end behavior

What changed is the architecture. AutoScientist is no longer just:
- ingestion -> graph -> dream -> consolidate

It is now more accurately:
- acquisition (`Reader`, `Researcher`, `Ingestor`)
- generative cognition (`Dreamer`, `Thinker`)
- adversarial scrutiny (`Critic`)
- delayed incubation (`InsightBuffer`)
- monitoring and mission tracking (`Observer`)
- test/execution (`Sandbox`)
- explanation and human interface (`Notebook`, `Conversation`, `GUI`)

That means the benchmark strategy must now evaluate not only output quality, but
also whether the newer cognitive control layers improve precision without
destroying novelty.

## Benchmark Rules

The revised benchmark should follow these rules across all implemented
dimensions:

- Every benchmark must state what level it measures:
  - `prompt-level` — isolated prompt behavior only
  - `module-level` — a real module method with its internal post-processing
  - `pipeline-level` — end-to-end behavior across multiple modules
- A pass criterion must fail on the metric's PRIMARY failure signal.
  A benchmark must not pass if its own summary shows the target behavior
  regressed on the main quantity being measured.
- Each metric should include:
  - positive cases
  - hard negatives
  - boundary / abstention checks
- LLM-as-judge prompts must inspect the actual generated evidence whenever
  possible, not just self-reported grades, counts, or metadata.
- If a benchmark cannot exercise the behavior it claims to test
  (for example: no deep insights, no deferred verdicts, no reinforced edges),
  it should report that explicitly and fail or mark the run as skipped /
  inconclusive. It must not fabricate dummy examples just to keep running.
- For regression testing, prefer deterministic or frozen fixtures over live
  network content. Live-search benchmarks are still useful, but they should
  evaluate the real filtered/persisted outputs of the system, not only raw
  retrieval.
- Skipped tests do not count as passes in aggregate reporting.

## Current Status

Implemented benchmark suites:
- `benchmark/dim1` - Knowledge Graph Quality
- `benchmark/dim2` - Dream Cycle Effectiveness
- `benchmark/dim3` - Thinker / Structured Reasoning Quality
- `benchmark/dim4` - Critic / System 2 Quality
- `benchmark/dim5` - Research & Reading Acquisition
- `benchmark/dim6` - Consolidation & Insight Buffer Quality

Not yet implemented, but now part of the revised roadmap:
- `dim7` - Observer & Mission Tracking
- `dim8` - Sandbox Hypothesis Testing
- `dim9` - Conversation & Groundedness
- `dim10` - End-to-End + Ablation Studies
- `dim11` - Technical / Operational Reliability
- `dim12` - Salience & Neuromodulatory Scheduling

## Dimension Overview

## D1 - Knowledge Graph Quality

Purpose:
Validate the substrate that every other module depends on.

Keep from the original plan:
- node quality
- duplicate rate
- edge accuracy
- cluster coherence
- contradiction detection

Add in the revised plan:
- graph/index consistency after ingestion, reading, and research
- mission-link calibration for newly added nodes
- provenance-aware analysis where useful (reading vs research vs conversation)

Interpretation:
D1 remains valid with only modest revision. It is still the most stable and
least architecture-sensitive benchmark dimension.

## D2 - Dream Cycle Effectiveness

Purpose:
Measure whether dreaming produces useful questions, valid insights, and mission
progress rather than just stylistic novelty.

Keep from the original plan:
- question quality
- insight validity
- mission advance calibration
- walk diversity
- NREM effectiveness (includes Hippocampal Replay evaluation)

Major revision required:
Dream outputs now need to be evaluated in two layers:
- raw dream outputs
- critic-reviewed / critic-accepted outputs

New D2 metrics should include:
- raw deep-insight validity
- critic-accepted deep-insight validity
- precision lift from critic review
- false-negative rate of critic review on valuable dream insights
- quality of dream insights deferred to the `InsightBuffer` and promoted later

Interpretation:
D2 is still valid, but the current suite should be treated as only the first
half of the Dreamer benchmark until the critic-aware slices are added.

## D3 - Thinker / Structured Reasoning Quality

Purpose:
Evaluate deliberate reasoning, decomposition, and answer-oriented thinking.

Core metrics:
- pattern selection accuracy
- reductive sub-question sufficiency
- insight actionability
- cross-round coherence
- mission-answer specificity improvement across thinking rounds
- usefulness of thinker-generated sub-questions for later research

Why it is now separate:
`Thinker` is no longer a side feature. It is a core cognitive module and should
not be hidden inside end-to-end tests.

## D4 - Critic / System 2 Quality

Purpose:
Measure whether the critic improves the system's claims rather than just adding
latency and friction.

Core metrics:
- review activation calibration
- verdict accuracy (`accept` / `refine` / `reject` / `defer`)
- novelty-check accuracy
- refinement quality lift
- defer quality
- bypass safety for low-stakes outputs

This is now essential because the critic changes both `Dreamer` and `Thinker`
behavior in meaningful ways.

## D5 - Research & Reading Acquisition

Purpose:
Measure the quality of external knowledge acquisition.

Core metrics:
- retrieval relevance
- extraction signal-to-noise ratio
- resolution rate for agenda questions
- reading list quality
- index freshness after ingestion
- predictive-processing calibration during ingestion
- deduplication accuracy for near-equivalent claims
- thinker-question vs observer-question retrieval quality

Interpretation:
D5 is no longer just "can the system fetch something relevant?" It must also
measure whether acquisition writes cleanly into the graph, whether surprise is
being used meaningfully, and whether different question generators retrieve
materially different quality of evidence.

## D6 - Consolidation & Insight Buffer Quality

Purpose:
Evaluate whether the system deepens and cleans its knowledge over time.

Core metrics:
- synthesis genuineness
- abstraction quality
- gap inference accuracy
- contradiction maintenance / unresolved-tension prioritization
- decay calibration
- delayed insight promotion quality
- time-to-promotion behavior for the insight buffer

Why the roadmap is extended here:
Consolidation is not only generative. A good evening pass should also preserve
and surface unresolved tensions so future cycles spend effort where the graph is
most conflicted. That maintenance behavior is distinct from D1 contradiction
detection, which measures whether contradictions are recognized in the first
place.

Evaluation note:
Each D6 metric should be tested with multiple positive cases, hard negatives,
and boundary-condition checks. In practice that means semantic metrics should
stress both abstention and specificity, while maintenance metrics should verify
selective behavior rather than only single happy-path examples.

## D7 - Observer & Mission Tracking

Purpose:
Measure whether self-monitoring highlights the right events.

Core metrics:
- emergence precision / recall
- question deduplication accuracy
- incubation-age calibration
- mission advance recall
- mission-link coverage on high-value nodes

## D8 - Sandbox Hypothesis Testing

Purpose:
Evaluate testability judgment, code generation, execution success, and graph
integration of empirical results.

Core metrics:
- testability classification accuracy
- code execution success rate
- verdict calibration against known ground truth cases
- graph integration quality
- whether critic/thinker improve the quality of hypotheses sent to sandbox

## D9 - Conversation & Groundedness

Purpose:
Evaluate the human-facing interface for faithfulness and useful interaction.

Core metrics:
- groundedness against graph content
- retrieved-node usefulness
- idea-ingestion fidelity
- multi-turn coherence
- running-hypothesis grounding

## D10 - End-to-End + Ablations

Purpose:
Measure the whole system over time and determine which modules matter.

Revised ablation set:
- full pipeline
- no Dreamer
- no Thinker
- no Critic
- no Dreamer + no Thinker
- naive ingestion baseline

Key metrics:
- knowledge accumulation trajectory
- mission progress over cycles
- self-correction rate
- mission suspension / resumption behavior

## D11 - Technical & Operational Reliability

Purpose:
Measure whether the system is operationally trustworthy at scale.

Core metrics:
- phase throughput vs graph size
- LLM call counts by phase
- RAM footprint
- persistence reliability
- concurrent-access stability
- recovery from interruption
- graph/index consistency after crash/restart
- GUI-path vs scheduler-path parity

## D12 - Salience & Neuromodulatory Scheduling

Purpose:
Measure whether the autonomous scheduler dynamically adapts to internal states using the neuromodulatory architecture.

Core metrics:
- dopamine response rate (does high dopamine correctly trigger focused thinking/research?)
- frustration response rate (does high frustration correctly trigger wandering dreams to escape local minima?)
- episodic record usefulness

## Recommended Evaluation Order

Given the current state of the repo, the practical implementation order should be:

1. Finish revising `dim1`
2. Finish revising `dim2`
3. Build `dim3` for `Thinker`
4. Build `dim4` for `Critic`
5. Build `dim10` ablations early, even before every other module suite exists
6. Fill in the remaining module-specific dimensions

Reason:
The largest new unknowns introduced by the recent architecture changes are:
- whether `Thinker` adds real value
- whether `Critic` improves precision enough to justify its cost
- whether the full system is better than simpler ablations

## Notes for the Existing Suites

### `dim1`
Treat the current implementation as still valid and worth keeping.
It maps cleanly to the revised roadmap.

### `dim2`
Treat the current implementation as partially complete.
It still measures raw dream quality well, but it does not fully measure the
new critic-mediated dream pipeline yet. The revised D2 target is:
- raw dream metrics
- post-critic dream metrics
- precision-lift analysis

## Reporting Structure

The final benchmark report should still preserve the original 3-tier structure:

- Tier 1 - Core claims from the README
- Tier 2 - Comparative performance vs ablations/baselines
- Tier 3 - Failure mode catalog

That structure still holds. The major revision is simply that `Thinker` and
`Critic` are now part of the core claims rather than optional extras.
