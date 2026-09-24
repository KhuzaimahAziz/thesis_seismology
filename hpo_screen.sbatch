#!/usr/bin/env bash
#SBATCH --job-name=phasenet_hpo
#SBATCH --array=0-9%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out

  set -euo pipefail

  cd /home/khuzaima/project/thesis_seismology

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate seisbench

  CONFIGS=(
    "scratch 3e-4 0"
    "scratch 3e-4 1e-4"
    "scratch 1e-3 0"
    "scratch 1e-3 1e-4"
    "transfer 1e-5 0"
    "transfer 1e-5 1e-5"
    "transfer 3e-5 0"
    "transfer 3e-5 1e-5"
    "transfer 1e-4 0"
    "transfer 1e-4 1e-5"
  )

  TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array job.}"

  read -r VARIANT LR WD <<< "${CONFIGS[$TASK_ID]}"

  if [[ "$VARIANT" == "scratch" ]]; then
    BASELINE=null
  else
    BASELINE=original
  fi

  echo "Hyperparameter trial: ${TASK_ID}"
  echo "Model variant: ${VARIANT}"
  echo "Learning rate: ${LR}"
  echo "Weight decay: ${WD}"
  echo "Conda environment: ${CONDA_DEFAULT_ENV}"

  srun train-seisbench \
    experiment_name=hyperparameter_screening \
    training.run_purpose=hyperparameter_screening \
    training.epochs=25 \
    training.lr="$LR" \
    training.weight_decay="$WD" \
    training.baseline_model_name="$BASELINE" \
    training.seed=42