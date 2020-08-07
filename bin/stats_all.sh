#!/bin/bash

# Check env
if [ -z "$TRACE_OUTDIR" ]; then 
    echo "TRACE_OUTDIR is unset"
    exit 1
fi

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

# Run stats
for trace_bench in $benchmarks; do
  TRACE_BENCH=$trace_bench bin/stats.sh
done