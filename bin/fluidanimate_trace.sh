#!/bin/bash

in_fluid="/opt/parsec-3.0/pkgs/apps/fluidanimate/run/in_35K.fluid"
out_fluid="/opt/parsec-3.0/pkgs/apps/fluidanimate/run/out.fluid"
cmd="/opt/parsec-3.0/pkgs/apps/fluidanimate/inst/-.gcc-serial/bin/fluidanimate 1 5 $in_fluid $out_fluid" # simsmall

docker run -d \
    --name fluidanimate \
    -v /media/disk/traces/fluidanimate:/out \
    -v /media/disk/parsec-3.0:/opt/parsec-3.0 \
    -w /out \
    mcapuccini/dl-prefetch sh -c \
    "pin -t pinatrace.so -- $cmd"