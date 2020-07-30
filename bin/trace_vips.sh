#!/bin/bash

# This tool does not have ROI marked
# https://lists.cs.princeton.edu/pipermail/parsec-users/2015-October/001681.html

# Set env
export TRACE_BENCH="vips"
input_base="/opt/parsec-3.0/pkgs/apps/vips/run"
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/vips/inst/amd64-linux.gcc-hooks/bin/vips im_benchmark $input_base/bigben_2662x5500.v $input_base/output.v" # simlarge

# Run
bin/trace.sh