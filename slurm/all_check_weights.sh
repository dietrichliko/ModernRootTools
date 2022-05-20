#!/bin/sh -x
#SBATCH --job-name="check_weights"
#SBATCH --output="/scratch-cbe/users/dietrich.liko/MRT/check_weights_%A_%a.log"
#SBATCH --time="01:00:00"
#SBATCH --qos="rapid"
#SBATCH --array="0-7"
#SBATCH --chdir="/users/dietrich.liko/working/MRT/ModernRootTools"


PERIODES=("Run2016preVFP" "Run2016postVFP" "Run2017" "Run2018")
SKIMS=("Met" "MetLepEnergy")

poetry run check_weights \
    --period ${PERIODES[$((SLURM_ARRAY_TASK_ID % 4))]} \
    --skim ${SKIMS[$((SLURM_ARRAY_TASK_ID / 4))]} \
    --log-level DEBUG \
    --root-threads 1
