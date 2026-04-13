# Dimension 5: Research & Reading Acquisition Benchmark Report

This report aggregates the Dimension 5 suite for retrieval relevance,
extraction quality, resolution rate, reading list quality, index
freshness, predictive processing, and dedup accuracy.

## Summary

**Overall Pipeline Status:** 8 / 8 Tests Passed

| Test Component | Status | Key Metric |
| -------------- | ------ | ---------- |
| Retrieval Relevance | PASS | questions_evaluated: 5 |
| Extraction SNR | PASS | passages_evaluated: 4 |
| Resolution Rate | PASS | questions_evaluated: 4 |
| Reading List Quality | PASS | total_suggestions: 8 |
| Index Freshness | PASS | new_nodes_created: 4 |
| Predictive Processing | PASS | aligned_mean_importance: 0.500 |
| Dedup Accuracy | PASS | base_node_count: 5 |
| Thinker vs Observer | PASS | thinker_questions_count: 4 |

## Detailed Results

### Retrieval Relevance
**Verdict:** PASS

- **questions_evaluated**: 5
- **total_results**: 20
- **mean_precision**: 0.917
- **mean_relevance**: 5.65

---

### Extraction SNR
**Verdict:** PASS

- **passages_evaluated**: 4
- **total_nodes_extracted**: 19
- **signal_count**: 18
- **signal_fraction**: 0.947
- **mean_quality**: 6.263

---

### Resolution Rate
**Verdict:** PASS

- **questions_evaluated**: 4
- **resolved_count**: 4
- **resolution_rate**: 1.0
- **mean_advancement**: 4.25

---

### Reading List Quality
**Verdict:** PASS

- **total_suggestions**: 8
- **mean_score**: 6.417
- **redundant_count**: 0

---

### Index Freshness
**Verdict:** PASS

- **new_nodes_created**: 4
- **self_retrieval_rate**: 1.0
- **paraphrase_recall**: 1.0
- **index_consistent**: True
- **index_size**: 4
- **graph_size**: 4

---

### Predictive Processing
**Verdict:** PASS

- **aligned_mean_importance**: 0.5
- **baseline_mean_importance**: 0.5
- **misaligned_mean_importance**: 0.8
- **full_ordering_correct**: False
- **basic_ordering_correct**: True

---

### Dedup Accuracy
**Verdict:** PASS

- **base_node_count**: 5
- **final_node_count**: 16
- **test_cases**: 8
- **correct_count**: 8
- **accuracy**: 1.0
- **merge_precision**: 1.0
- **merge_recall**: 1.0

---

### Thinker vs Observer
**Verdict:** PASS

- **thinker_questions_count**: 4
- **observer_questions_count**: 3
- **thinker_precision**: 0.944
- **thinker_mean_relevance**: 6.322
- **thinker_mean_mission_value**: 5.978
- **observer_precision**: 0.889
- **observer_mean_relevance**: 6.467
- **observer_mean_mission_value**: 6.111
- **relevance_advantage**: -0.145
- **mission_advantage**: -0.133

---
