#!/usr/bin/env bash
# Dimension 4: Critic / System 2 Quality Benchmark Runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

JUDGE_MODEL="llama3.1:70b"
RESULTS_DIR="benchmark/dim4/results"

while [[ $# -gt 0 ]]; do
  case $1 in
    --judge-model)
      JUDGE_MODEL="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

echo "================================================================="
echo "  AutoScientist Benchmark - Dimension 4: Critic / System 2"
echo "  Judge Model: $JUDGE_MODEL"
echo "================================================================="
echo ""

echo "--> [1/6] Running D4 Test 1: Activation Calibration..."
python benchmark/dim4/test_d4_activation_calibration.py \
    --out "$RESULTS_DIR/d4_activation_calibration.json"
echo ""

echo "--> [2/6] Running D4 Test 2: Verdict Accuracy..."
python benchmark/dim4/test_d4_verdict_accuracy.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d4_verdict_accuracy.json"
echo ""

echo "--> [3/6] Running D4 Test 3: Novelty Check Accuracy..."
python benchmark/dim4/test_d4_novelty_check.py \
    --out "$RESULTS_DIR/d4_novelty_check.json"
echo ""

echo "--> [4/6] Running D4 Test 4: Refinement Quality Lift..."
python benchmark/dim4/test_d4_refinement_lift.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d4_refinement_lift.json"
echo ""

echo "--> [5/6] Running D4 Test 5: Defer Quality..."
python benchmark/dim4/test_d4_defer_quality.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d4_defer_quality.json"
echo ""

echo "--> [6/6] Running D4 Test 6: Bypass Safety..."
python benchmark/dim4/test_d4_bypass_safety.py \
    --out "$RESULTS_DIR/d4_bypass_safety.json"
echo ""

echo "--> Generating aggregate report..."
python benchmark/dim4/report_d4.py

echo ""
echo "================================================================="
echo "  Dimension 4 Benchmark Complete."
echo "  Summary available in: benchmark/dim4/results/report_d4_summary.md"
echo "================================================================="
