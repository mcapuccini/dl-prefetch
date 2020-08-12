#!/bin/bash

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi

# Args
if [ $# -lt 1 ]; then
    echo "Not enough arguments"
    exit 1
fi
script_cmd="${@:1}"

# Benchmarks
benchmarks="
  blackscholes
  bodytrack
  canneal
  dedup
  facesim
  ferret
  fluidanimate
  freqmine
  raytrace
  streamcluster
  swaptions
  vips
  x264
"

# Run script
for trace_bench in $benchmarks; do
  TRACE_BENCH=$trace_bench bin/script.sh $script_cmd
done