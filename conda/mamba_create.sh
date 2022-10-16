#!/bin/sh -x

if [ "$CONDA_DEFAULT_ENV" != "my-base" ]
then
    conda activate my-base
fi

mamba create -y -n mrt -c conda-forge --file=pkgs-root626-py310.txt
