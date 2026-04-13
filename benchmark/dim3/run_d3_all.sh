#!/usr/bin/env bash
# Dimension 3: Thinker / Structured Reasoning Benchmark Runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

JUDGE_MODEL="llama3.1:70b"
RESULTS_DIR="benchmark/dim3/results"

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
echo "  AutoScientist Benchmark - Dimension 3: Thinker"
echo "  Judge Model: $JUDGE_MODEL"
echo "================================================================="
echo ""

echo "--> [1/6] Running D3 Test 1: Pattern Selection Accuracy..."
python benchmark/dim3/test_d3_pattern_selection.py \
    --out "$RESULTS_DIR/d3_pattern_selection.json"
echo ""

echo "--> [2/6] Running D3 Test 2: Reductive Sub-question Sufficiency..."
python benchmark/dim3/test_d3_reductive_sufficiency.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d3_reductive_sufficiency.json"
echo ""

echo "--> [3/6] Running D3 Test 3: Insight Actionability..."
python benchmark/dim3/test_d3_actionability.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d3_actionability.json"
echo ""

echo "--> [4/6] Running D3 Test 4: Cross-round Coherence..."
python benchmark/dim3/test_d3_cross_round_coherence.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d3_cross_round_coherence.json"
echo ""

echo "--> [5/6] Running D3 Test 5: Mission-answer Specificity Lift..."
python benchmark/dim3/test_d3_specificity_lift.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d3_specificity_lift.json"
echo ""

echo "--> [6/6] Running D3 Test 6: Sub-question Utility..."
python benchmark/dim3/test_d3_subquestion_utility.py \
    --judge-model "$JUDGE_MODEL" \
    --out "$RESULTS_DIR/d3_subquestion_utility.json"
echo ""

echo "--> Generating aggregate report..."
python benchmark/dim3/report_d3.py

echo ""
echo "================================================================="
echo "  Dimension 3 Benchmark Complete."
echo "  Summary available in: benchmark/dim3/results/report_d3_summary.md"
echo "================================================================="
