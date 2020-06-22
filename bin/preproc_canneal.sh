#!/bin/bash

# Set env
export TRACE_BENCH="canneal"
export TRACE_CMD="/opt/parsec-3.0/pkgs/kernels/canneal/inst/amd64-linux.gcc-serial/bin/canneal 1 15000 2000 /opt/parsec-3.0/pkgs/kernels/canneal/run/400000.nets 128" # simlarge
# TRACE_CMD="/opt/parsec-3.0/pkgs/kernels/canneal/inst/amd64-linux.gcc-serial/bin/canneal 1 10000 2000 /opt/parsec-3.0/pkgs/kernels/canneal/run/100000.nets 32" # simsmall

# Run
bin/preproc.sh