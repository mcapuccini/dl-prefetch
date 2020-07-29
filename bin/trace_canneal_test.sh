#!/bin/bash

# Set env
export TRACE_BENCH="canneal_test"
export TRACE_CMD="/opt/parsec-3.0/pkgs/kernels/canneal/inst/amd64-linux.gcc-serial/bin/canneal 1 5 100 /opt/parsec-3.0/pkgs/kernels/canneal/run/10.nets 1" # test

# Run
bin/trace.sh