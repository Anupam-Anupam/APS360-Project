#!/bin/bash
# Check output for a specific job

JOB_ID=${1:-22371400}

echo "=== Job Status ==="
sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,StdOut,StdErr

echo ""
echo "=== Output File ==="
OUTPUT_FILE=$(sacct -j $JOB_ID --format=StdOut --noheader --parsable2 | head -1)
if [ -n "$OUTPUT_FILE" ] && [ "$OUTPUT_FILE" != "None" ]; then
    echo "Output file: $OUTPUT_FILE"
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "=== Last 100 lines of output ==="
        tail -100 "$OUTPUT_FILE"
    else
        echo "Output file not found at: $OUTPUT_FILE"
        echo ""
        echo "Trying common locations:"
        ls -lh ~/MaCroScope/*${JOB_ID}*.out 2>/dev/null || echo "Not found in ~/MaCroScope/"
        ls -lh $SCRATCH/MaCroScope/*${JOB_ID}*.out 2>/dev/null || echo "Not found in \$SCRATCH/MaCroScope/"
    fi
else
    echo "No output file path found. Trying common locations:"
    ls -lh ~/MaCroScope/*${JOB_ID}*.out 2>/dev/null
    ls -lh $SCRATCH/MaCroScope/*${JOB_ID}*.out 2>/dev/null
fi

echo ""
echo "=== Error File ==="
ERROR_FILE=$(sacct -j $JOB_ID --format=StdErr --noheader --parsable2 | head -1)
if [ -n "$ERROR_FILE" ] && [ "$ERROR_FILE" != "None" ]; then
    echo "Error file: $ERROR_FILE"
    if [ -f "$ERROR_FILE" ]; then
        echo ""
        echo "=== Last 100 lines of errors ==="
        tail -100 "$ERROR_FILE"
    else
        echo "Error file not found at: $ERROR_FILE"
        echo ""
        echo "Trying common locations:"
        ls -lh ~/MaCroScope/*${JOB_ID}*.err 2>/dev/null || echo "Not found in ~/MaCroScope/"
        ls -lh $SCRATCH/MaCroScope/*${JOB_ID}*.err 2>/dev/null || echo "Not found in \$SCRATCH/MaCroScope/"
    fi
else
    echo "No error file path found. Trying common locations:"
    ls -lh ~/MaCroScope/*${JOB_ID}*.err 2>/dev/null
    ls -lh $SCRATCH/MaCroScope/*${JOB_ID}*.err 2>/dev/null
fi
