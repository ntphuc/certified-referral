#!/bin/bash
#SBATCH --job-name=certref_ens
#SBATCH --output=logs/ens_%j.out
#SBATCH --error=logs/ens_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2g.35gb:1
# Ensemble (K=3 seeds) disagreement scores, inference only. See ensemble_scores.py.
set -euo pipefail
mkdir -p logs
PY="${PY:-python}"
: "${BRISC_ROOT:?set BRISC_ROOT}" "${KVASIR_ROOT:?set KVASIR_ROOT}"
$PY -u ensemble_scores.py --dataset kvasir --runs outputs_kvasir --data-root "$KVASIR_ROOT"
$PY -u ensemble_scores.py --dataset brisc --runs outputs --data-root "$BRISC_ROOT"
echo ENS_DONE
