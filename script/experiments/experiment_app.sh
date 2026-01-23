#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
DATA_ROOT="$PROJECT_ROOT/MaxSAT_Dataset"
OUTPUT_DIR="$PROJECT_ROOT/solver_output"
MODEL_DIR="$PROJECT_ROOT/plots/SGAT/Godel"
declare -a arr=("2020" "2021" "2022" "2023" "2024")
seed=${1:-1}
sgat_id=${2:-1}
cuda_device=${3:-0}

mkdir -p \
  "$OUTPUT_DIR/LM" \
  "$OUTPUT_DIR/ModelPredict"

# Run classical solver (lm) for timeout 60
timeout=60
suf=""


result_exists() {
  local csv_path="$1"
  local problem_path="$2"

  if [ ! -f "$csv_path" ]; then
    return 1
  fi

  python - "$csv_path" "$problem_path" <<'PY'
import csv
import sys
csv_path, problem_path = sys.argv[1:]

from pathlib import Path

try:
  with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    target_basename = Path(problem_path).name
    for row in reader:
      p = row.get("problem")
      if not p:
        continue
      # exact match or basename match
      if p == problem_path or Path(p).name == target_basename:
        sys.exit(0)
except FileNotFoundError:
  pass

sys.exit(1)
PY
}

for year in "${arr[@]}"; do
  directory="$DATA_ROOT/maxsat${year}_sel"

  if [ ! -d "$directory" ]; then
    echo "Skipping missing directory: $directory" >&2
    continue
  fi

  # Prepare classical output
  out_lm="$OUTPUT_DIR/LM/results_${year}_${seed}${suf}.csv"
  out_model_pred="$OUTPUT_DIR/ModelPredict/results_${sgat_id}_${year}_${seed}${suf}.csv"

  # Iterate files safely
  while IFS= read -r -d '' file; do
    if [ -f "$file" ]; then

      # Some .wcnf files lack a 'p wcnf' header line; since we're already iterating
      # over *.wcnf files, run the weight-detection unconditionally so clause
      # weights (e.g. 71) are always detected. Look for clause lines that start
      # with a positive integer weight > 1. Allow optional leading whitespace.
      if awk 'BEGIN{w=0} /^[[:space:]]*[0-9]+[[:space:]]/{ if($1+0>1){w=1; exit} } END{exit !w}' "$file"; then
        continue
      fi
        # Classical lm solver (unweighted evaluation) - skip if already present
        if ! result_exists "$out_lm" "$file"; then
          python "$SRC_DIR/solve.py" \
            --solver lm \
            --problem "$file" \
            --timeout "$timeout" \
            --seed "$seed" \
            --save-cost-path "$out_lm"
        else
          echo "Skipping lm for $file (already present in $out_lm)"
        fi

        # Model-only prediction solver to measure raw GNN assignment accuracy - skip if already present
        if ! result_exists "$out_model_pred" "$file"; then
          python "$SRC_DIR/solve.py" \
            --solver model-predict \
            --problem "$file" \
            --timeout "$timeout" \
            --seed "$seed" \
            --model-dir "$MODEL_DIR" \
            --model-id "$sgat_id" \
            --cuda "$cuda_device" \
            --save-cost-path "$out_model_pred"
        else
          echo "Skipping model-predict for $file (already present in $out_model_pred)"
        fi
    fi
  done < <(find "$directory" -type f -name '*.wcnf' -print0)
done
