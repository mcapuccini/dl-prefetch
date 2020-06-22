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

# Helper
function run_autocorr {
    subset=$1
    docker run -d \
        --name "${TRACE_BENCH}_${subset}_autocorr" \
        -v "${TRACE_OUTDIR}/${TRACE_BENCH}":/trace \
        mcapuccini/dl-prefetch \
        python scripts/autocorrelation.py \
        --trace-path /trace/${subset}.feather \
        --out-path /trace/autocor_${subset}.npy
} 

# Run
run_autocorr "dev"
docker logs "${TRACE_BENCH}_dev_autocorr"
run_autocorr "test"
docker logs "${TRACE_BENCH}_test_autocorr"
run_autocorr "train"