#!/bin/sh -x

for env in mrtools-root622-py39 mrtools-root624-py39 mrtools-root624-py310
do
    conda remove -y -n $env --all
done
