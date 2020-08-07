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

# Run stats
docker run -d \
  --name ${TRACE_BENCH}_stats \
  -v ${TRACE_OUTDIR}:/traces \
  mcapuccini/dl-prefetch sh -c \
  "python scripts/stats.py --dataset-dir /traces/${TRACE_BENCH}"