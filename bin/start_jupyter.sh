#!/bin/bash

cmd="jupyter notebook --ip=0.0.0.0 --NotebookApp.token='abcde' --allow-root --no-browser"
docker run -d \
    --name jupyter \
    -v /media/disk/traces:/traces \
    -p 8888:8888 \
    -p 4040:4040 \
    mcapuccini/dl-prefetch \
    $cmd