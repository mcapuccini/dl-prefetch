#!/bin/bash

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi

# Run roitrace
docker run -d \
    --name "preprocess_many" \
    -v ${TRACE_OUTDIR}:/traces \
    -e ARROW_PRE_0_15_IPC_FORMAT=1 \
    mcapuccini/dl-prefetch \
    python scripts/preprocess_many.py --traces-path /traces