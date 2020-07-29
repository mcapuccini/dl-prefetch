#!/bin/bash

# Set env
export TRACE_BENCH="ferret"
input_base="/opt/parsec-3.0/pkgs/apps/ferret/run"
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/ferret/inst/amd64-linux.gcc-hooks/bin/ferret $input_base/corel lsh $input_base/queries 10 20 1 $input_base/output.txt" # simmedium, gcc-hooks

# Run
bin/trace.sh