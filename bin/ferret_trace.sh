#!/bin/bash

corel="/opt/parsec-3.0/pkgs/apps/ferret/run/corel"
queries="/opt/parsec-3.0/pkgs/apps/ferret/run/queries"
output="/opt/parsec-3.0/pkgs/apps/ferret/run/output.txt"
cmd="/opt/parsec-3.0/pkgs/apps/ferret/inst/-.gcc-serial/bin/ferret $corel lsh $queries 10 20 1 $output" # simsmall

docker run -d \
    --name ferret \
    -v /media/disk/traces/ferret:/out \
    -v /media/disk/parsec-3.0:/opt/parsec-3.0 \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t pinatrace.so -- $cmd"