#!/bin/bash

# Set env
export TRACE_BENCH="facesim"
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/facesim/inst/amd64-linux.gcc-serial/bin/facesim -timing -threads 1" # simlarge

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi
if [ -z "$TRACE_PARSECDIR" ]; then 
    echo "TRACE_PARSECDIR is unset"
    exit 1
fi

# Run roitrace (output in /opt/parsec-3.0/pkgs/apps/facesim/run/)
docker run -d \
    --name ${TRACE_BENCH}_facesim \
    -v $TRACE_PARSECDIR:/opt/parsec-3.0 \
    -e LD_LIBRARY_PATH=/opt/parsec-3.0/pkgs/libs/hooks/inst/amd64-linux.gcc-serial/lib \
    --security-opt seccomp=unconfined \
    -w /opt/parsec-3.0/pkgs/apps/facesim/run \
    mcapuccini/dl-prefetch sh -c \
    "pin -t roitrace.so -- $TRACE_CMD"