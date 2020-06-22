#!/bin/bash

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi
if [ -z "$TRACE_BENCH" ]; then 
    echo "TRACE_BENCH is unset"
    exit 1
fi

# Run
docker run -d \
    --name ${TRACE_BENCH}_preproc \
    -v ${TRACE_OUTDIR}:/traces \
    -e ARROW_PRE_0_15_IPC_FORMAT=1 \
    mcapuccini/dl-prefetch \
    python scripts/preprocessing.py \
    --trace-path /traces/${TRACE_BENCH}/roitrace.csv \
    --out-dir /traces/${TRACE_BENCH}