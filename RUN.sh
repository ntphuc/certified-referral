#!/usr/bin/env bash
# One array task = one (width, seed) run, then a refresh of the shared analysis.
# All paths and settings live here; run_uit.slurm only holds SBATCH settings.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATASET="${DATASET:-brisc}"            # brisc | kvasir
if [[ "$DATASET" == "kvasir" ]]; then
  : "${DATA_ROOT:?set DATA_ROOT to the Kvasir-SEG folder (images/, masks/)}"
  OUT_DIR="${OUT_DIR:-outputs_kvasir}"; RESULTS_DIR="${RESULTS_DIR:-results_kvasir}"
  MIN_COV="${MIN_COV:-0.3}"
else
  : "${DATA_ROOT:?set DATA_ROOT to the extracted BRISC2025 folder}"
fi
MIN_COV="${MIN_COV:-0.2}"
PY="${PY:-python}"                     # python with torch, opencv-python, pandas, matplotlib
MANIFEST="data/brisc2025_manifest.csv"
WIDTHS=(${WIDTHS:-4 8 16 32 64})
SEEDS=(${SEEDS:-2026 2027 2028})
EPOCHS="${EPOCHS:-60}"
OUT_DIR="${OUT_DIR:-outputs}"          # smoke test: OUT_DIR=outputs_smoke
RESULTS_DIR="${RESULTS_DIR:-results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"           # smoke test: EXTRA_ARGS="--limit 64"

IDX="${SLURM_ARRAY_TASK_ID:-${IDX:-0}}"
NW=${#WIDTHS[@]}
WIDTH=${WIDTHS[$((IDX % NW))]}
SEED=${SEEDS[$((IDX / NW))]}

mkdir -p logs data cache "$OUT_DIR" "$RESULTS_DIR"
export PYTHONUNBUFFERED=1
[[ -d "$DATA_ROOT" ]] || { echo "[FATAL] dataset missing: $DATA_ROOT" >&2; exit 2; }
"$PY" -c "import torch, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
command -v nvidia-smi >/dev/null && nvidia-smi || true
echo "[$(date '+%F %T')] dataset=$DATASET task=$IDX width=$WIDTH seed=$SEED epochs=$EPOCHS"

# Manifest + caches: built once, other array tasks wait on the lock.
(
  flock 9
  if [[ "$DATASET" == "brisc" && ! -f "$MANIFEST" ]]; then
    "$PY" -u scripts/prepare_brisc2025.py --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
      --prefer-backend raw
  fi
  "$PY" -u prepare_cache.py --dataset "$DATASET" --manifest "$MANIFEST" --data-root "$DATA_ROOT" --cache-dir cache
) 9>data/.prep.lock

"$PY" -u train_eval.py --dataset "$DATASET" --width "$WIDTH" --seed "$SEED" --epochs "$EPOCHS" \
  --manifest "$MANIFEST" --data-root "$DATA_ROOT" --cache-dir cache --out "$OUT_DIR" $EXTRA_ARGS

# Refresh the pooled analysis with whatever runs have finished so far.
(
  flock 8
  "$PY" -u analyze.py --runs "$OUT_DIR" --out "$RESULTS_DIR" --min-coverage "$MIN_COV"
) 8>"$RESULTS_DIR/.analysis.lock" || echo "[warn] analysis failed (results incomplete?)"
echo "[$(date '+%F %T')] task $IDX done"
