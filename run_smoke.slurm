#!/bin/bash
#SBATCH --job-name=certref_smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2g.35gb:1
# Quick end-to-end check: width 8, 64 samples per split, 2 epochs, separate output dirs.
# Also builds the full-size data cache that the real array job reuses.
mkdir -p logs
IDX=1 EPOCHS=2 OUT_DIR=outputs_smoke RESULTS_DIR=results_smoke EXTRA_ARGS="--limit 64 --mc-passes 3" bash RUN.sh
