#!/bin/bash

# Set env
export TRACE_BENCH="raytrace"
input_base="/opt/parsec-3.0/pkgs/apps/raytrace/run"
# export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/raytrace/inst/amd64-linux.gcc-serial/bin/rtview $input_base/happy_buddha.obj -automove -nthreads 1 -frames 3 -res 1920 1080" # simlarge
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/raytrace/inst/amd64-linux.gcc-serial/bin/rtview $input_base/happy_buddha.obj -automove -nthreads 1 -frames 3 -res 480 270" # simsmall

# Run
bin/trace.sh