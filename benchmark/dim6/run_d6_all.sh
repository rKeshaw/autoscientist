#!/usr/bin/env bash
# Dimension 6: Consolidation & Insight Buffer — Run all tests
set -e

cd "$(dirname "$0")/../.."

MODEL="gemma4:latest"
JUDGE_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            shift
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --judge-model=*)
            JUDGE_MODEL="${1#*=}"
            shift
            ;;
        *)
            if [[ "$1" != --* ]] && [[ "$MODEL" == "gemma4:latest" ]]; then
                MODEL="$1"
                shift
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$JUDGE_MODEL" ]]; then
    JUDGE_MODEL="$MODEL"
fi

echo "=================================================="
echo "  Dimension 6: Consolidation & Insight Buffer"
echo "  Model: ${MODEL}"
echo "  Judge model: ${JUDGE_MODEL}"
echo "=================================================="

mkdir -p benchmark/dim6/results

if [[ ! -f benchmark/dim4/shared/brain.json ]]; then
    echo "Dim4 shared graph not found. Run benchmark/dim4/prep_d4_graph.py first." >&2
    exit 1
fi

echo ""
echo "── T1: Synthesis Genuineness ──"
python benchmark/dim6/test_d6_synthesis_genuineness.py \
    --model "${MODEL}" \
    --judge-model "${JUDGE_MODEL}"

echo ""
echo "── T2: Abstraction Quality ──"
python benchmark/dim6/test_d6_abstraction_quality.py \
    --model "${MODEL}" \
    --judge-model "${JUDGE_MODEL}"

echo ""
echo "── T3: Gap Inference Accuracy ──"
python benchmark/dim6/test_d6_gap_inference.py \
    --model "${MODEL}" \
    --judge-model "${JUDGE_MODEL}"

echo ""
echo "── T4: Contradiction Maintenance ──"
python benchmark/dim6/test_d6_contradiction_maintenance.py

echo ""
echo "── T5: Decay Calibration ──"
python benchmark/dim6/test_d6_decay_calibration.py

echo ""
echo "── T6: Delayed Insight Promotion ──"
python benchmark/dim6/test_d6_delayed_insight_promotion.py

echo ""
echo "── T7: Time-to-Promotion ──"
python benchmark/dim6/test_d6_time_to_promotion.py

echo ""
echo "── Generating Report ──"
python benchmark/dim6/report_d6.py

echo ""
echo "Done. Report at: benchmark/dim6/results/report_d6_summary.md"
