#!/bin/bash
# On Fir: show output for test jobs 23679792 and 23682342.
# Run from login node: bash show_fir_job_output.sh
# (Or scp this to Fir and run it there.)

cd "${SCRATCH:-/scratch/anupam}/MaCroScope" || exit 1

for JOB in 23679792 23682342; do
  echo "========== Job $JOB =========="
  sacct -j "$JOB" --format=JobID,JobName,State,ExitCode,Start,End -n
  echo ""
  FOUND=
  for F in "logs/test-$JOB.out" "logs/test-129-$JOB.out" "slurm-$JOB.out"; do
    if [ -f "$F" ]; then
      echo "--- $F (last 120 lines) ---"
      tail -120 "$F"
      FOUND=1
      echo ""
    fi
  done
  if [ -z "$FOUND" ]; then
    echo "No output file found (tried logs/test-$JOB.out, logs/test-129-$JOB.out, slurm-$JOB.out)"
  fi
  echo ""
done
