#!/bin/bash

# Set env
export TRACE_BENCH="blackscholes"
input_base="/opt/parsec-3.0/pkgs/apps/blackscholes/run"
# export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.gcc-serial/bin/blackscholes 1 $input_base/in_64K.txt $input_base/prices.txt" # simlarge
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/blackscholes/inst/amd64-linux.gcc-serial/bin/blackscholes 1 $input_base/in_4K.txt $input_base/prices.txt" # simsmall

# Run
bin/trace.sh