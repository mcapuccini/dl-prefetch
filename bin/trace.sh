#!/bin/bash

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi
if [ -z "$TRACE_PARSECDIR" ]; then 
    echo "TRACE_PARSECDIR is unset"
    exit 1
fi
if [ -z "$TRACE_CMD" ]; then 
    echo "TRACE_CMD is unset"
    exit 1
fi
if [ -z "$TRACE_BENCH" ]; then 
    echo "TRACE_BENCH is unset"
    exit 1
fi

# Run roitrace
docker run -d \
    --name $TRACE_BENCH \
    -v ${TRACE_OUTDIR}/${TRACE_BENCH}:/out \
    -v $TRACE_PARSECDIR:/opt/parsec-3.0 \
    -e LD_LIBRARY_PATH=/opt/parsec-3.0/pkgs/libs/hooks/inst/amd64-linux.gcc-serial/lib \
    --security-opt seccomp=unconfined \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t roitrace.so -- $TRACE_CMD"