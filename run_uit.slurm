#!/bin/bash
#SBATCH --job-name=certref_brisc
#SBATCH --output=logs/certref_%A_%a.out
#SBATCH --error=logs/certref_%A_%a.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --array=0-14
# 15 tasks = 5 widths (4 8 16 32 64) x 3 seeds (2026 2027 2028); see RUN.sh.
# Rerun a single task:  sbatch --array=7 run_uit.slurm
# Finished runs (outputs/w*_s*/summary.json) are skipped automatically.
mkdir -p logs
bash RUN.sh
