# Setup development environment

[Mamba](https://github.com/mamba-org/mamba) is a reimplementation of the conda package manager in C++ recommended for the use with conda-forge and large installations.

Create __my-base__ virtual environment to bootstrap _mamba_.

    conda env create -f env-my-base.yaml

Create __mrtools-root624-py310__ virtual environment. It covers the installation
of packages, that cannot be installed with _poetry_ (or _pip_)

    conda activate my-base
    mamba create -y -n mrtools-root624-py310 -c conda-forge --file=pkgs-root624-py310.txt
    conda deactivate my-base

Install __MRTools__ and its associated python packages. _poetry_ provides better control
of python packages. As _conda_ and _poetry_ are not interoperable, the environment has to be recreated from scratch for modifications.

    conda activate mrtools-root624-py310
    poetry install
