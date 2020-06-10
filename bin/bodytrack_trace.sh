#!/bin/bash

cmd="/opt/parsec-3.0/pkgs/apps/bodytrack/inst/-.gcc-serial/bin/bodytrack /opt/parsec-3.0/pkgs/apps/bodytrack/run/sequenceB_1 4 1 1000 5 0 1" # simsmall
# cmd="/opt/parsec-3.0/pkgs/apps/bodytrack/inst/-.gcc-serial/bin/bodytrack /opt/parsec-3.0/pkgs/apps/bodytrack/run/sequenceB_1 4 1 5 1 0 1" # test

docker run -d \
    --name bodytrack \
    -v /media/disk/traces/bodytrack:/out \
    -v /media/disk/parsec-3.0:/opt/parsec-3.0 \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t pinatrace.so -- $cmd"