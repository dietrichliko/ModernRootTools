#!/bin/sh -x

for name in root622-py39 root624-py39 root624-py310
do
    mamba create -y -n "mrtools-$name" -c conda-forge --file="pkgs-$name.txt"
done
