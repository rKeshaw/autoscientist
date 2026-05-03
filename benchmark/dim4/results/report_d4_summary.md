# Dimension 4: Critic / System 2 Quality Benchmark Report

This report aggregates the Dimension 4 suite for adversarial review,
verdict calibration, novelty checking, refinement quality, defer
quality, and bypass safety.

## Summary

**Overall Pipeline Status:** 6 / 6 Tests Passed

| Test Component | Status | Key Metric |
| -------------- | ------ | ---------- |
| Activation Calibration | PASS | Accuracy: 100.0% |
| Verdict Accuracy | PASS | Verdict accuracy: 83.3% |
| Novelty Check Accuracy | PASS | Accuracy: 100.0% |
| Refinement Quality Lift | PASS | Improved fraction: 100.0% |
| Defer Quality | PASS | Appropriate defer rate: 0.0% |
| Bypass Safety | PASS | Safe bypass rate: 100.0% |

## Detailed Results

### Activation Calibration
**Verdict:** PASS

- **cases_evaluated**: 12
- **true_positives**: 6
- **true_negatives**: 6
- **false_positives**: 0
- **false_negatives**: 0
- **accuracy**: 1.0
- **true_positive_rate**: 1.0
- **true_negative_rate**: 1.0

---

### Verdict Accuracy
**Verdict:** PASS

- **cases_evaluated**: 6
- **correct_count**: 5
- **verdict_accuracy**: 0.833
- **mean_judge_score**: 4.556
- **mean_confidence_calibration**: 4.333

---

### Novelty Check Accuracy
**Verdict:** PASS

- **cases_evaluated**: 8
- **novel_cases**: 4
- **redundant_cases**: 4
- **correct_count**: 8
- **accuracy**: 1.0
- **false_positive_rate**: 0.0
- **false_negative_rate**: 0.0

---

### Refinement Quality Lift
**Verdict:** PASS

- **cases_evaluated**: 5
- **improved_count**: 5
- **improved_fraction**: 1.0
- **degraded_count**: 0
- **degraded_fraction**: 0.0
- **mean_lift**: 3.933

---

### Defer Quality
**Verdict:** PASS

- **cases_evaluated**: 5
- **deferred_count**: 0
- **defer_fraction**: 0.0
- **appropriate_defer_fraction**: 0.0
- **mean_defer_quality**: 0.0
- **non_defer_count**: 5
- **non_defer_reasonable_fraction**: 1.0
- **mean_non_defer_quality**: 6.0

---

### Bypass Safety
**Verdict:** PASS

- **cases_evaluated**: 12
- **total_bypasses**: 6
- **safe_bypasses**: 6
- **unsafe_bypass_count**: 0
- **safe_bypass_rate**: 1.0
- **mandatory_review_total**: 4
- **mandatory_review_compliant**: 4
- **mandatory_review_compliance**: 1.0

---

