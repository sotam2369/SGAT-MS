#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
TOOLS_DIR="$PROJECT_ROOT/tools"
DATA_ROOT="$PROJECT_ROOT/MaxSAT_Dataset"
PLOTS_DIR="$PROJECT_ROOT/plots"
OUTPUT_DIR="$PROJECT_ROOT/solver_output"
HYBRID_OUTPUT_DIR="$OUTPUT_DIR/hybrid_solver"
NUWLS_DIR="$PROJECT_ROOT/NuWLS/NuWLS-source-code_hybrid"
BANDHS_DIR="$PROJECT_ROOT/BandHS/BandHS-main_sgat"
SATLIKE_DIR="$PROJECT_ROOT/SATLike3.0_sgat"

declare -a arr=("2020" "2021" "2022" "2023")
seed=${1:-1}
gat_id=${2:-1}
sgat_id=${3:-1}
cuda_device=${4:-0}
# legacy `id` used by results_formatter -> keep pointing at sgat_id for compatibility
id="$sgat_id"

mkdir -p \
  "$PLOTS_DIR/GAT" \
  "$PLOTS_DIR/SGAT" \
  "$HYBRID_OUTPUT_DIR/GAT/NuWLS" \
  "$HYBRID_OUTPUT_DIR/GAT/BandHS" \
  "$HYBRID_OUTPUT_DIR/GAT/SATLike3.0" \
  "$HYBRID_OUTPUT_DIR/SGAT/NuWLS" \
  "$HYBRID_OUTPUT_DIR/SGAT/BandHS" \
  "$HYBRID_OUTPUT_DIR/SGAT/SATLike3.0" \
  "$OUTPUT_DIR/NuWLS" \
  "$OUTPUT_DIR/BandHS" \
  "$OUTPUT_DIR/SATLike3.0" \
  "$OUTPUT_DIR/fig"

# Unified loops: run hybrid (GAT and SGAT) and classical solvers for timeouts 60 and 300
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

    # Prepare hybrid output files (SGAT) - now under hybrid_solver/SGAT
    hybrid_out_nuwls="$HYBRID_OUTPUT_DIR/SGAT/NuWLS/results_${sgat_id}_${year}_${seed}${suf}.csv"
    hybrid_out_bandhs="$HYBRID_OUTPUT_DIR/SGAT/BandHS/results_${sgat_id}_${year}_${seed}${suf}.csv"
    hybrid_out_satlike="$HYBRID_OUTPUT_DIR/SGAT/SATLike3.0/results_${sgat_id}_${year}_${seed}${suf}.csv"

    # Prepare hybrid output files (GAT) - placed under HYBRID_OUTPUT_DIR/GAT
    gat_hybrid_out_nuwls="$HYBRID_OUTPUT_DIR/GAT/NuWLS/results_${gat_id}_${year}_${seed}${suf}.csv"
    gat_hybrid_out_bandhs="$HYBRID_OUTPUT_DIR/GAT/BandHS/results_${gat_id}_${year}_${seed}${suf}.csv"
    gat_hybrid_out_satlike="$HYBRID_OUTPUT_DIR/GAT/SATLike3.0/results_${gat_id}_${year}_${seed}${suf}.csv"

    # Prepare classical outputs
    out_nuwls="$OUTPUT_DIR/NuWLS/results_${year}_${seed}${suf}.csv"
    out_bandhs="$OUTPUT_DIR/BandHS/results_${year}_${seed}${suf}.csv"
    out_satlike="$OUTPUT_DIR/SATLike3.0/results_${year}_${seed}${suf}.csv"

    # Iterate files safely
    while IFS= read -r -d '' file; do
      if [ -f "$file" ]; then
        # Hybrid w/ SGAT model (write into hybrid_solver/SGAT/...)
        python "$SRC_DIR/solve.py" \
          --solver nuwls \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$NUWLS_DIR" \
          --init \
          --model-dir "$PLOTS_DIR/SGAT/" \
          --model-id "$sgat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$hybrid_out_nuwls"

        python "$SRC_DIR/solve.py" \
          --solver bandhs \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$BANDHS_DIR" \
          --init \
          --model-dir "$PLOTS_DIR/SGAT/" \
          --model-id "$sgat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$hybrid_out_bandhs"

        python "$SRC_DIR/solve.py" \
          --solver satlike3.0 \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$SATLIKE_DIR" \
          --init \
          --model-dir "$PLOTS_DIR/SGAT/" \
          --model-id "$sgat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$hybrid_out_satlike"

        # Hybrid w/ GAT model (write into hybrid_solver/GAT/...)
        python "$SRC_DIR/solve.py" \
          --solver nuwls \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$NUWLS_DIR" \
          --init \
          --model-dir "$PLOTS_DIR/GAT/" \
          --model-id "$gat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$gat_hybrid_out_nuwls"

        python "$SRC_DIR/solve.py" \
          --solver bandhs \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$BANDHS_DIR" \
          --init \
          --model-dir "$PLOTS_DIR/GAT/" \
          --model-id "$gat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$gat_hybrid_out_bandhs"

        python "$SRC_DIR/solve.py" \
          --solver satlike3.0 \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$SATLIKE_DIR" \
          --init \
          --model-dir "$PLOTS_DIR/GAT/" \
          --model-id "$gat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$gat_hybrid_out_satlike"

        # Classical solvers
        python "$SRC_DIR/solve.py" \
          --solver nuwls \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$NUWLS_DIR" \
          --save-cost-path "$out_nuwls"

        python "$SRC_DIR/solve.py" \
          --solver bandhs \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$BANDHS_DIR" \
          --save-cost-path "$out_bandhs"

        python "$SRC_DIR/solve.py" \
          --solver satlike3.0 \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --solver-dir "$SATLIKE_DIR" \
          --save-cost-path "$out_satlike"
      fi
    done < <(find "$directory" -type f -name '*.wcnf' -print0)
  done
done

# Keep variables used by results_formatter compatible with prior layout (sgat hybrid results)
nuwls="NuWLS/results_year_${seed}.csv"
hybrid_nuwls="hybrid_solver/SGAT/NuWLS/results_${id}_year_${seed}.csv"
bandhs="BandHS/results_year_${seed}.csv"
hybrid_bandhs="hybrid_solver/SGAT/BandHS/results_${id}_year_${seed}.csv"
satlike="SATLike3.0/results_year_${seed}.csv"
hybrid_satlike="hybrid_solver/SGAT/SATLike3.0/results_${id}_year_${seed}.csv"

python "$TOOLS_DIR/results_formatter.py" \
  "$OUTPUT_DIR" \
  "$OUTPUT_DIR/fig" \
  --files "$nuwls" "$hybrid_nuwls" "$bandhs" "$hybrid_bandhs" "$satlike" "$hybrid_satlike" \
  --years 2020 2021 2022 2023 \
  --names NuWLS "w/ SGAT" BandHS "w/ SGAT" SATLike3.0 "w/ SGAT" \
  --best_cost "$PROJECT_ROOT/vba/vbayear_uw.csv" \
  --show_scores \
  --proposed_method 1 3 \
  --show-all
