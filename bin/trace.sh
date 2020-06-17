#!/bin/bash

# TRACE_CMD (canneal)
TRACE_CMD="/opt/parsec-3.0/pkgs/kernels/canneal/inst/amd64-linux.gcc-serial/bin/canneal 1 10000 2000 /opt/parsec-3.0/pkgs/kernels/canneal/run/100000.nets 32" # simsmall

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

# Run roitrace
docker run -d \
    --name $(basename $TRACE_OUTDIR) \
    -v $TRACE_OUTDIR:/out \
    -v $TRACE_PARSECDIR:/opt/parsec-3.0 \
    -e LD_LIBRARY_PATH=/opt/parsec-3.0/pkgs/libs/hooks/inst/amd64-linux.gcc-serial/lib \
    --security-opt seccomp=unconfined \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t roitrace.so -- $TRACE_CMD"