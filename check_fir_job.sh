#!/bin/bash
# Script to check progress and logs of fir tinker task
# Usage: Run this on the fir cluster login node

JOB_ID=20739127
SCRATCH_DIR="$SCRATCH/MaCroScope"

echo "=== Job Status ==="
squeue -j $JOB_ID

echo ""
echo "=== Recent Output Log (last 50 lines) ==="
if [ -f "$SCRATCH_DIR/tinker-train-$JOB_ID.out" ]; then
    tail -50 "$SCRATCH_DIR/tinker-train-$JOB_ID.out"
else
    echo "Output log not found at $SCRATCH_DIR/tinker-train-$JOB_ID.out"
fi

echo ""
echo "=== Recent Error Log (last 50 lines) ==="
if [ -f "$SCRATCH_DIR/tinker-train-$JOB_ID.err" ]; then
    tail -50 "$SCRATCH_DIR/tinker-train-$JOB_ID.err"
else
    echo "Error log not found at $SCRATCH_DIR/tinker-train-$JOB_ID.err"
fi

echo ""
echo "=== Training Log Directory ==="
if [ -d "$SCRATCH_DIR/log" ]; then
    ls -lh "$SCRATCH_DIR/log/"
    echo ""
    echo "=== Checkpoint File ==="
    if [ -f "$SCRATCH_DIR/log/checkpoints.jsonl" ]; then
        cat "$SCRATCH_DIR/log/checkpoints.jsonl"
    else
        echo "Checkpoint file not found"
    fi
else
    echo "Log directory not found at $SCRATCH_DIR/log"
fi

echo ""
echo "=== To follow logs in real-time, run: ==="
echo "tail -f $SCRATCH_DIR/tinker-train-$JOB_ID.out"
