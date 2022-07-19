#!/bin/sh -x

if [ "$CONDA_DEFAULT_ENV" != "my-base" ]
then
    conda activate my-base
fi

for env in $(conda info --envs | grep -oe "^mrtools-root[[:alnum:]-]*")
do
    conda remove -y -n $env --all
done


for name in root626-py310
do
    mamba create -y -n "mrt-$name" -c conda-forge --file="pkgs-$name.txt"
done
