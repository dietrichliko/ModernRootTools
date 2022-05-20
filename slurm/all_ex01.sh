#!/bin/bash -x
#SBATCH --job-name="example01"
#SBATCH --output="/scratch-cbe/users/dietrich.liko/MRT/example01_%A_%a.log"
#SBATCH --time="01:00:00"
#SBATCH --qos="rapid"
#SBATCH --cpus-per-task=10
#SBATCH --array="0-11"
#SBATCH --chdir="/users/dietrich.liko/working/MRT/ModernRootTools"

PERIODS=("Run2016preVFP" "Run2016postVFP" "Run2017" "Run2018")
OPTS=("" "--no-tight" "--trigger")
NAMES=("ex01_{period}" "ex01_{period}_medium" "ex01_{period}_trigger")

poetry run example01 \
    --period ${PERIODS[$((SLURM_ARRAY_TASK_ID % 4))]} \ 
    --name ${NAMES[$((SLURM_ARRAY_TASK_ID / 4))]} \
    ${OPTS[$((SLURM_ARRAY_TASK_ID / 4))]} \
    --log-level DEBUG

poetry run ex01_plot01  \
  --period ${PERIODS[$((SLURM_ARRAY_TASK_ID % 4))]} \
  --name  ${NAMES[$((SLURM_ARRAY_TASK_ID / 4))]}  \
  --log-level DEBUG 
  


