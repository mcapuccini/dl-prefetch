#!/bin/bash

# Set env
export TRACE_BENCH="fluidanimate"
input_base="/opt/parsec-3.0/pkgs/apps/fluidanimate/run"
# export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/fluidanimate/inst/amd64-linux.gcc-serial/bin/fluidanimate 1 5 $input_base/in_300K.fluid $input_base/out.fluid" # simlarge
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/fluidanimate/inst/amd64-linux.gcc-serial/bin/fluidanimate 1 5 $input_base/in_35K.fluid $input_base/out.fluid" # simsmall

# Run
bin/trace.sh