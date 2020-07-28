#!/bin/bash

# Set env
export TRACE_BENCH="canneal_test"

# Run
bin/preproc.sh --spark-driver-memory 4G --spark-driver-max-result-size 2G --test-size 1000