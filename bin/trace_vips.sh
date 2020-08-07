#!/bin/bash

# This bench does not have ROI marked in version 3, using 2.1
# https://lists.cs.princeton.edu/pipermail/parsec-users/2015-October/001681.html

# Set env
export TRACE_BENCH="vips"
input_base="/opt/parsec-2.1/pkgs/apps/vips/run"
# export TRACE_CMD="/opt/parsec-2.1/pkgs/apps/vips/inst/amd64-linux.gcc-serial/bin/vips im_benchmark $input_base/vulture_2336x2336.v $input_base/output.v" # simmedium
export TRACE_CMD="/opt/parsec-2.1/pkgs/apps/vips/inst/amd64-linux.gcc-serial/bin/vips im_benchmark $input_base/pomegranate_1600x1200.v $input_base/output.v" # simsmall

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi
if [ -z "$TRACE_PARSECDIR_2" ]; then 
    echo "TRACE_PARSECDIR_2 is unset"
    exit 1
fi

# Run roitrace
docker run -d \
    --name ${TRACE_BENCH}_trace \
    -v ${TRACE_OUTDIR}/${TRACE_BENCH}:/out \
    -v $TRACE_PARSECDIR_2:/opt/parsec-2.1 \
    -e LD_LIBRARY_PATH=/opt/parsec-2.1/pkgs/libs/hooks/inst/amd64-linux.gcc-serial/lib \
    --security-opt seccomp=unconfined \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t roitrace.so -- $TRACE_CMD"