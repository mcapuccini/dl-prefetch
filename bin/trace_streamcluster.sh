#!/bin/bash

# Set env
export TRACE_BENCH="streamcluster"
input_base="/opt/parsec-3.0/pkgs/apps/fluidanimate/run"
# export TRACE_CMD="/opt/parsec-3.0/pkgs/kernels/streamcluster/inst/amd64-linux.gcc-serial/bin/streamcluster 10 20 64 8192 8192 1000 none $input_base/output.txt 1" # simmedium
export TRACE_CMD="/opt/parsec-3.0/pkgs/kernels/streamcluster/inst/amd64-linux.gcc-serial/bin/streamcluster 10 20 32 4096 4096 1000 none $input_base/output.txt 1" # simsmall

# Run
bin/trace.sh