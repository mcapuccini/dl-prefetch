#!/bin/bash

dataset="$1"
docker run -d \
    --name "${dataset}-autocorr" \
    -v /media/disk/traces/${dataset}:/trace \
    mcapuccini/dl-prefetch \
    python scripts/autocorrelation.py \
    --trace-path /trace/pinatrace.out \
    --out-path /trace/autocorr.npy