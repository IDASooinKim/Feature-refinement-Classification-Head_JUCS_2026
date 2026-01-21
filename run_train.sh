#!/bin/bash

# ===========================================================
# run.sh : Training launcher with log saving
#
# Usage:
#   bash run.sh                # single GPU / CPU
#   bash run.sh ddp 4          # run DDP with 4 GPUs
#
# Log file is automatically saved to:
#   logs/run_<timestamp>.log
# ===========================================================

set -e

# Create logs directory
mkdir -p logs

# Timestamp for log file name
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="logs/run_${TIMESTAMP}.log"

echo "[run.sh] Log file: $LOG_FILE"

MODE=$1
NUM_PROCS=$2

# -----------------------------------------------------------
# Single GPU / CPU mode
# -----------------------------------------------------------
if [ "$MODE" = "" ]; then
    echo "[run.sh] Running in single-process mode..."
    python main.py 2>&1 | tee "$LOG_FILE"
    exit 0
fi

# -----------------------------------------------------------
# Distributed Data Parallel Mode (DDP)
# -----------------------------------------------------------
if [ "$MODE" = "ddp" ]; then
    if [ "$NUM_PROCS" = "" ]; then
        echo "[run.sh] ERROR: GPU count required."
        echo "Usage: bash run.sh ddp 4"
        exit 1
    fi

    echo "[run.sh] Running in DDP mode with $NUM_PROCS GPUs..."
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_PROCS \
        main.py 2>&1 | tee "$LOG_FILE"

    exit 0
fi

echo "[run.sh] Unknown mode: $MODE"
echo "Usage:"
echo "  bash run.sh"
echo "  bash run.sh ddp 4"
exit 1
