#!/bin/bash

prices="/opt/parsec-3.0/pkgs/apps/blackscholes/run/prices.txt"
in_4K="/opt/parsec-3.0/pkgs/apps/blackscholes/run/in_4K.txt"
cmd="/opt/parsec-3.0/pkgs/apps/blackscholes/inst/-.gcc-serial/bin/blackscholes 1 $in_4K $prices" # simsmall

docker run -d \
    --name blackscholes \
    -v /media/disk/traces/blackscholes:/out \
    -v /media/disk/parsec-3.0:/opt/parsec-3.0 \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t pinatrace.so -- $cmd"