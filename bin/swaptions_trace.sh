#!/bin/bash

cmd="/opt/parsec-3.0/pkgs/apps/swaptions/inst/-.gcc-serial/bin/swaptions -ns 16 -sm 10000 -nt 1" # simsmall

docker run -d \
    --name swaptions \
    -v /media/disk/traces/swaptions:/out \
    -v /media/disk/parsec-3.0:/opt/parsec-3.0 \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t pinatrace.so -- $cmd"