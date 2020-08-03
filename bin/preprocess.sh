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
    mcapuccini/dl-prefetch \
    python scripts/preprocess_many.py --traces-path /traces