#!/bin/bash

docker run -d \
    --name canneal_test_trace \
    -v /media/disk/traces/canneal_test:/out \
    -v /media/disk/parsec-3.0:/opt/parsec-3.0 \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    'pin -t pinatrace.so -- /opt/parsec-3.0/pkgs/kernels/canneal/inst/-.gcc-serial/bin/canneal 1 5 100 10.nets 1'