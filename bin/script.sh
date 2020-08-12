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

# Args
if [ $# -lt 1 ]; then
    echo "Not enough arguments"
    exit 1
fi
script_cmd="${@:1}"

# Run stats
docker run -d \
  --name ${script_cmd%% *}_${TRACE_BENCH} \
  -v ${TRACE_OUTDIR}:/traces \
  mcapuccini/dl-prefetch sh -c \
  "python scripts/$script_cmd --dataset-dir /traces/${TRACE_BENCH}"