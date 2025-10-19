#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
DATA_ROOT="$PROJECT_ROOT/MaxSAT_Dataset"
PLOTS_DIR="$PROJECT_ROOT/plots"
OUTPUT_DIR="$PROJECT_ROOT/solver_output"
MIXING_DIR="$PROJECT_ROOT/mixing"
MIXSAT_DIR="$PROJECT_ROOT/mixsat"
FOURIER_DIR="$PROJECT_ROOT/FourierSAT"

declare -a arr=("2020" "2021" "2022" "2023" "2024")
seed=${1:-1}
gat_id=${2:-1}
sgat_id=${3:-1}
cuda_device=${4:-0}

mkdir -p \
  "$PLOTS_DIR/GAT" \
  "$PLOTS_DIR/SGAT" \
  "$OUTPUT_DIR/LS-GAT" \
  "$OUTPUT_DIR/LS-SGAT" \
  "$OUTPUT_DIR/MIXSAT" \
  "$OUTPUT_DIR/Mixing" \
  "$OUTPUT_DIR/FourierSAT"

# Unified loop: run all solvers per file for each year, for timeout=60 and timeout=300
for timeout in 60 300; do
  if [ "$timeout" -eq 300 ]; then
    suf="_300s"
  else
    suf="_60s"
  fi

  for year in "${arr[@]}"; do
    directory="$DATA_ROOT/maxsat${year}_sel"

    if [ ! -d "$directory" ]; then
      echo "Skipping missing directory: $directory" >&2
      continue
    fi

    # Use find with -print0 to handle filenames with spaces/newlines
    while IFS= read -r -d '' file; do
      # LS-GAT (uses GAT model directory and gat_id)
      python "$SRC_DIR/solve.py" \
        --solver sgat \
        --train \
        --problem "$file" \
        --timeout "$timeout" \
        --seed "$seed" \
        --model-dir "$PLOTS_DIR/GAT/" \
        --model-id "$gat_id" \
        --cuda "$cuda_device" \
        --save-cost-path "$OUTPUT_DIR/LS-GAT/results_${gat_id}_${year}_${seed}${suf}.csv"

      # LS-SGAT (uses SGAT model directory and sgat_id)
      python "$SRC_DIR/solve.py" \
        --solver sgat \
        --train \
        --problem "$file" \
        --timeout "$timeout" \
        --seed "$seed" \
        --model-dir "$PLOTS_DIR/SGAT/Godel" \
        --model-id "$sgat_id" \
        --cuda "$cuda_device" \
        --save-cost-path "$OUTPUT_DIR/LS-SGAT/results_${sgat_id}_${year}_${seed}${suf}.csv"

      # MIXSAT
      python "$SRC_DIR/solve.py" \
        --solver mixsat \
        --problem "$file" \
        --timeout "$timeout" \
        --seed "$seed" \
        --solver-dir "$MIXSAT_DIR" \
        --save-cost-path "$OUTPUT_DIR/MIXSAT/results_${year}_${seed}${suf}.csv"

      # Mixing
      python "$SRC_DIR/solve.py" \
        --solver mixing \
        --problem "$file" \
        --timeout "$timeout" \
        --seed "$seed" \
        --solver-dir "$MIXING_DIR" \
        --save-cost-path "$OUTPUT_DIR/Mixing/results_${year}_${seed}${suf}.csv"

      # FourierSAT
      python "$SRC_DIR/solve.py" \
        --solver fouriersat \
        --problem "$file" \
        --timeout "$timeout" \
        --seed "$seed" \
        --solver-dir "$FOURIER_DIR" \
        --save-cost-path "$OUTPUT_DIR/FourierSAT/results_${year}_${seed}${suf}.csv"

    done < <(find "$directory" -type f -name '*.wcnf' -print0)
  done
done
