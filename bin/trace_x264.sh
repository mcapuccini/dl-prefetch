#!/bin/bash

# Set env
export TRACE_BENCH="x264"
input_base="/opt/parsec-3.0/pkgs/apps/x264/run"
# this command crashes (but it does it after ROI ends)
export TRACE_CMD="/opt/parsec-3.0/pkgs/apps/x264/inst/amd64-linux.gcc-serial/bin/x264 --quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 --threads 1 -o $input_base/eledream.264 $input_base/eledream_640x360_32.y4m" # simmedium

# Run
bin/trace.sh