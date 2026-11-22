#!/bin/bash
#SBATCH --job-name=certref_kvasir
#SBATCH --output=logs/kvasir_%A_%a.out
#SBATCH --error=logs/kvasir_%A_%a.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --array=0-14
# Kvasir-SEG: same 15 width x seed tasks; outputs_kvasir/, results_kvasir/.
# Rerun a single task:  sbatch --array=7 run_kvasir.slurm
# Finished runs (outputs/w*_s*/summary.json) are skipped automatically.
mkdir -p logs
DATASET=kvasir bash RUN.sh
