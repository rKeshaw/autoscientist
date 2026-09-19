#!/bin/bash
set -e

MISSION="How do critical phase transitions in physical systems relate to edge-of-chaos dynamics and computational capacity in biological and artificial neural networks?"

echo "=========================================================="
echo "Starting Fresh Run on New Mission:"
echo "$MISSION"
echo "Start time: $(date)"
echo "=========================================================="

mkdir -p data logs

echo ""
echo "--- [Stage 1/2] Running Bootstrap ---"
PYTHONPATH=. .venv/bin/python bootstrap.py "$MISSION" 2>&1 | tee logs/bootstrap.log

echo ""
echo "--- [Stage 2/2] Running 2-Loop Cycle Scheduler ---"
PYTHONPATH=. .venv/bin/python scheduler/scheduler.py --mode cycle --loops 2 2>&1 | tee logs/scheduler.log

echo ""
echo "=========================================================="
echo "Fresh 2-Loop Run Complete!"
echo "End time: $(date)"
echo "=========================================================="
