#!/bin/bash

# Set env
export TRACE_BENCH="swaptions"
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/swaptions/inst/amd64-linux.gcc-serial/bin/swaptions -ns 32 -sm 20000 -nt 1" # simmedium

# Run
bin/trace.sh