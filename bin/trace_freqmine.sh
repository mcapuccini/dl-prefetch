#!/bin/bash

# Set env
export TRACE_BENCH="freqmine"
input_base="/opt/parsec-3.0/pkgs/apps/freqmine/run"
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/freqmine/inst/amd64-linux.gcc-serial/bin/freqmine $input_base/kosarak_500k.dat 410" # simmedium

# Run
bin/trace.sh