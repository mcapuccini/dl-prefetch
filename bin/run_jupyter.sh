#!/bin/bash

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi
if [ -z "$JUPYTER_TOKEN" ]; then 
    echo "JUPYTER_TOKEN is unset"
    exit 1
fi

# Run Jupyter
docker run -d \
    --name jupyter \
    -v ${TRACE_OUTDIR}:/traces \
    -p 8888:8888 \
    -p 4040:4040 \
    -e ARROW_PRE_0_15_IPC_FORMAT=1 \
    mcapuccini/dl-prefetch \
    jupyter notebook --no-browser \
    --allow-root \
    --port=8888 \
    --ip="0.0.0.0" \
    --NotebookApp.token="$JUPYTER_TOKEN"