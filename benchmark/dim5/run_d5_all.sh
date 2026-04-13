#!/usr/bin/env bash
# Dimension 5: Research & Reading Acquisition — Run all tests
set -e

cd "$(dirname "$0")/../.."

JUDGE_MODEL="${1:-llama3.1:70b}"
SKIP_NETWORK=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --judge-model=*) JUDGE_MODEL="${arg#*=}" ;;
        --judge-model)   shift; JUDGE_MODEL="${1}" ;;
        --skip-network)  SKIP_NETWORK="--skip-network" ;;
    esac
done

# Handle --judge-model as positional
if [[ "$1" != --* ]] && [[ -n "$1" ]]; then
    JUDGE_MODEL="$1"
fi

echo "=================================================="
echo "  Dimension 5: Research & Reading Acquisition"
echo "  Judge model: ${JUDGE_MODEL}"
echo "  Skip network: ${SKIP_NETWORK:-no}"
echo "=================================================="

mkdir -p benchmark/dim5/results

echo ""
echo "── T1: Retrieval Relevance ──"
python benchmark/dim5/test_d5_retrieval_relevance.py \
    --judge-model "${JUDGE_MODEL}" ${SKIP_NETWORK}

echo ""
echo "── T2: Extraction SNR ──"
python benchmark/dim5/test_d5_extraction_snr.py \
    --judge-model "${JUDGE_MODEL}"

echo ""
echo "── T3: Resolution Rate ──"
python benchmark/dim5/test_d5_resolution_rate.py \
    --judge-model "${JUDGE_MODEL}" ${SKIP_NETWORK}

echo ""
echo "── T4: Reading List Quality ──"
python benchmark/dim5/test_d5_reading_list.py \
    --judge-model "${JUDGE_MODEL}"

echo ""
echo "── T5: Index Freshness ──"
python benchmark/dim5/test_d5_index_freshness.py

echo ""
echo "── T6: Predictive Processing ──"
python benchmark/dim5/test_d5_predictive_processing.py

echo ""
echo "── T7: Dedup Accuracy ──"
python benchmark/dim5/test_d5_dedup_accuracy.py

echo ""
echo "── T8: Thinker vs Observer Retrieval ──"
python benchmark/dim5/test_d5_thinker_vs_observer.py \
    --judge-model "${JUDGE_MODEL}" ${SKIP_NETWORK}

echo ""
echo "── Generating Report ──"
python benchmark/dim5/gen_report_d5.py

echo ""
echo "Done. Report at: benchmark/dim5/results/report_d5_summary.md"
