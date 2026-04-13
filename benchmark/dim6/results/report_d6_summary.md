# Dimension 6: Consolidation & Insight Buffer Benchmark Report

This report aggregates the Dimension 6 suite for synthesis,
abstraction, gap inference, contradiction upkeep, decay, and delayed
insight promotion.

## Summary

**Overall Pipeline Status:** 3 / 7 Tests Passed

| Test Component | Status | Key Metric |
| -------------- | ------ | ---------- |
| Synthesis Genuineness | FAIL | Pos genuine: 0.0% | Incoherent abstain: 100.0% | Boundary safe: 100.0% |
| Abstraction Quality | FAIL | Pos pass: 0.0% | Neg pass: 100.0% |
| Gap Inference Accuracy | FAIL | Pos pass: 75.0% | Neg pass: 66.7% |
| Contradiction Maintenance | PASS | Updated contradictions: 13 |
| Decay Calibration | PASS | Pruned weak edge: True | Medium edge calibrated: True |
| Delayed Insight Promotion | FAIL | Promoted: 0 |
| Time-to-Promotion | PASS | Pruned: 1 |

## Detailed Results

### Synthesis Genuineness
**Verdict:** FAIL

- **positive_total**: 4
- **positive_genuine**: 0
- **positive_genuine_rate**: 0.0
- **positive_quality**: 0.0
- **incoherent_negative_total**: 4
- **incoherent_negative_abstained**: 4
- **incoherent_negative_abstain_rate**: 1.0
- **boundary_total**: 1
- **boundary_safe**: 1
- **boundary_safe_rate**: 1.0

---

### Abstraction Quality
**Verdict:** FAIL

- **suite_count**: 6
- **positive_suite_count**: 3
- **negative_suite_count**: 3
- **positive_pass_rate**: 0.0
- **negative_pass_rate**: 1.0
- **quality**: 6.833

---

### Gap Inference Accuracy
**Verdict:** FAIL

- **suite_count**: 7
- **positive_suite_count**: 4
- **negative_suite_count**: 3
- **positive_pass_rate**: 0.75
- **negative_pass_rate**: 0.667
- **quality**: 7.0

---

### Contradiction Maintenance
**Verdict:** PASS

- **contradictory_node_a_bumped**: True
- **contradictory_node_b_bumped**: True
- **control_node_unchanged**: True
- **capped_node_a_capped_at_one**: True
- **capped_node_b_capped_at_one**: True
- **contradictions_updated**: 13

---

### Decay Calibration
**Verdict:** PASS

- **weak_edge_pruned**: True
- **contradiction_preserved**: True
- **decay_exempt_edge_preserved**: True
- **medium_edge_decayed_but_preserved**: True
- **fresh_node_preserved**: True
- **stale_node_decayed**: True
- **edges_decayed_count**: 11
- **timestamp_file_updated**: True

---

### Delayed Insight Promotion
**Verdict:** FAIL

- **promoted_count**: 0
- **remaining_count**: 2
- **strong_pair_shared_neighbors**: 0
- **weak_pair_shared_neighbors**: 0
- **strong_pair_promoted**: False
- **weak_pair_left_pending**: True

---

### Time-to-Promotion
**Verdict:** PASS

- **pruned_count**: 1
- **remaining_count**: 1
- **pair_pruned**: True
- **younger_pair_still_pending**: True

---

