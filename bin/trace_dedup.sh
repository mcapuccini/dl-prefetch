#!/bin/bash

# Set env
export TRACE_BENCH="dedup"
input_base="/opt/parsec-3.0/pkgs/kernels/dedup/run"
export TRACE_CMD="/opt/parsec-3.0/pkgs/kernels/dedup/inst/amd64-linux.gcc-serial/bin/dedup -c -p -v -t 1 -i $input_base/media.dat -o $input_base/output.dat.ddp" # simmall

# Run
bin/trace.sh