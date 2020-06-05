#!/bin/bash

# cmd="/opt/parsec-3.0/pkgs/kernels/canneal/inst/-.gcc-serial/bin/canneal 1 5 100 /opt/parsec-3.0/pkgs/kernels/canneal/run/10.nets 1" # test
cmd="/opt/parsec-3.0/pkgs/kernels/canneal/inst/-.gcc-serial/bin/canneal 1 10000 2000 /opt/parsec-3.0/pkgs/kernels/canneal/run/100000.nets 32" # simsmall

docker run -d \
    --name canneal \
    -v /media/disk/traces/canneal:/out \
    -v /media/disk/parsec-3.0:/opt/parsec-3.0 \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t pinatrace.so -- $cmd"