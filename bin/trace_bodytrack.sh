#!/bin/bash

# Set env
export TRACE_BENCH="bodytrack"
input_base="/opt/parsec-3.0/pkgs/apps/bodytrack/run"
# export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/bodytrack/inst/amd64-linux.gcc-serial/bin/bodytrack $input_base/sequenceB_4 4 4 4000 5 0 1" # simlarge
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/bodytrack/inst/amd64-linux.gcc-serial/bin/bodytrack $input_base/sequenceB_1 4 1 1000 5 0 1" # simsmall

# Run
bin/trace.sh