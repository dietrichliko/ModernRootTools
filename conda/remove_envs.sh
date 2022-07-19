#!/bin/sh -x

for env in mrtools-root626-py310
do
    conda remove -y -n $env --all
done
