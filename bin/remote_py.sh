#!/bin/bash

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi

# Args
if [ $# -lt 1 ]; then
    echo "Not enough arguments"
    exit 1
fi
script_path="$1"
script_args="${@:2}"

# Run remote python script
docker run -i \
    -v ${TRACE_OUTDIR}:/traces \
    mcapuccini/dl-prefetch \
    python -u - $script_args < $script_path