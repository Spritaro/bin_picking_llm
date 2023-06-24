#!/usr/bin/env bash

docker run --rm -it \
    -v $PWD:/home/appuser/bin_picking_llm \
    bin_picking_llm bash
