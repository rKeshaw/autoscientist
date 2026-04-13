# Dimension 2: Dream Cycle Effectiveness Benchmark Report

This report aggregates the revised Dimension 2 suite, including
the original Dreamer-quality tests plus the critic-aware and
insight-buffer-aware extensions added by the new roadmap.

## Summary

**Overall Pipeline Status:** 5 / 7 Tests Passed

| Test Component | Status | Key Metric |
| -------------- | ------ | ---------- |
| Question Quality | PASS | Dream composite score: 14.0 |
| Insight Validity | FAIL | Raw validity: 45.1% | depth mix: S=91 I=0 |
| Mission Advance Precision | PASS | System/judge corr: 0.88 |
| Walk Diversity | PASS | Coverage: 95.7% |
| NREM Effectiveness | PASS | Replay overlap: 0.296 > 0.0 |
| Critic Precision Lift | FAIL | Accepted validity: 88.9% (lift +0.000, watered-down=2) |
| Deferred Insight Promotion Quality | PASS | Promotion rate: 100.0% | Genuine promotions: 100.0% |

## Detailed Results

### Question Quality
**Verdict:** PASS

- **num_questions_evaluated_each**: 15
- **dream_questions_generated**: 15
- **benchmark_exercised**: True
- **dream_scores**:
  - specificity: 4.93
  - answerability: 4.13
  - novelty: 4.93
  - total: 14.0
- **baseline_scores**:
  - specificity: 3.4
  - answerability: 2.73
  - novelty: 5.0
  - total: 11.13

---

### Insight Validity
**Verdict:** FAIL

- **num_deep_insights**: 91
- **genuine_count**: 41
- **genuine_fraction**: 0.451
- **raw_depth_counts**:
  - structural: 91
- **depth_counts**:
  - structural: 91
- **structural_count**: 91
- **isomorphism_count**: 0
- **benchmark_exercised**: True
- **benchmark_exercised_structural**: True
- **benchmark_exercised_isomorphism**: False
- **benchmark_exercised_depth_mix**: False

---

### Mission Advance Precision
**Verdict:** PASS

- **samples_evaluated**: 20
- **correlation**: 0.884
- **flagging_accuracy**: 0.9

---

### Walk Diversity
**Verdict:** PASS

- **total_nodes**: 47
- **total_steps_taken**: 80
- **unique_nodes_visited**: 45
- **coverage_fraction**: 0.957
- **mean_revisit_rate**: 1.778
- **visited_more_than_once**: 21

---

### NREM Effectiveness
**Verdict:** PASS

- **candidate_edges**: 68
- **candidate_replay_path_edges**: 2
- **reinforced_edges_evaluated**: 54
- **unreinforced_edges_evaluated**: 14
- **mean_replay_overlap_reinforced**: 0.296
- **mean_replay_overlap_unreinforced**: 0.0
- **replay_touch_fraction_reinforced**: 0.259
- **replay_touch_fraction_unreinforced**: 0.0
- **replay_path_fraction_reinforced**: 0.037
- **replay_path_fraction_unreinforced**: 0.0
- **benchmark_exercised**: True

---

### Critic Precision Lift
**Verdict:** FAIL

- **raw_deep_insights**: 9
- **raw_quality_count**: 8
- **raw_genuine_count**: 8
- **raw_validity**: 0.889
- **accepted_count**: 9
- **accepted_quality_count**: 8
- **accepted_genuine_count**: 8
- **accepted_validity**: 0.889
- **precision_lift**: 0.0
- **false_negative_rate**: 0.0
- **verdict_breakdown**:
  - accept: 9
  - refine: 0
  - reject: 0
  - defer: 0
- **material_revision_count**: 8
- **critic_action_count**: 8
- **benchmark_exercised**: True
- **thesis_preserved_revision_count**: 6
- **watered_down_accepts**: 2

---

### Deferred Insight Promotion Quality
**Verdict:** PASS

- **deferred_pairs**: 3
- **promoted_pairs**: 3
- **promotion_rate**: 1.0
- **genuine_promotions**: 3
- **genuine_fraction**: 1.0
- **mean_promotion_cycle**: 1.0
- **remaining_buffer_size**: 0

---

