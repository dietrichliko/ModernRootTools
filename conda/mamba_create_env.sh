#!/bin/bash -x
#
# Setup environment with Mamba

NAME=${1:-"mrt2"}

# expose functuions to shell
MAMBA_BASE=$(mamba info --base)
. $MAMBA_BASE/etc/profile.d/conda.sh
. $MAMBA_BASE/etc/profile.d/mamba.sh


mamba create -y -n "$NAME" python=3.11 root=6.28.0 git cmake poetry

conda activate "$NAME"
conda list --export > environment.lock

# https://github.com/python-poetry/poetry/issues/6408
find $CONDA_PREFIX/lib/python3.11/site-packages -name direct_url.json -delete
