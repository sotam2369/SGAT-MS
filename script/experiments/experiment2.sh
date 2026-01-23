#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
DATA_ROOT="$PROJECT_ROOT/MaxSAT_Dataset"
PLOTS_DIR="$PROJECT_ROOT/plots"
OUTPUT_DIR="$PROJECT_ROOT/solver_output"
MIXING_DIR="$PROJECT_ROOT/../mixing"
MIXSAT_DIR="$PROJECT_ROOT/../mixsat"
FOURIER_DIR="$PROJECT_ROOT/../FourierSAT"

declare -a arr=("2020" "2021" "2022" "2023" "2024")
declare -a processed_years=()
seed=${1:-1}
gat_id=${2:-1}
sgat_id=${3:-1}
cuda_device=${4:-0}

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

try:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("problem") == problem_path:
                sys.exit(0)
except FileNotFoundError:
    pass

sys.exit(1)
PY
}

run_results_formatter() {
  local timeout="$1"
  shift
  local years=("$@")
  # Optional last parameter: weighted suffix (e.g. _weighted) or empty string
  local weighted_suffix=""
  if [ "${#years[@]}" -ge 1 ]; then
    # If the caller passed an extra arg for weighted_suffix, it will be the last element
    # and we can detect if it starts with '_' to treat it as the suffix.
    last_index=$((${#years[@]} - 1))
    maybe_suffix="${years[$last_index]}"
    if [[ "$maybe_suffix" == _* ]]; then
      weighted_suffix="$maybe_suffix"
      unset 'years[$last_index]'
      # re-index array
      years=("${years[@]}")
    fi
  fi

  if [ "${#years[@]}" -eq 0 ]; then
    return
  fi

  local best_cost_file="$PROJECT_ROOT/vba/vbayear_uw.csv"

  local suf fig_dir
  if [ "$timeout" -eq 300 ]; then
    suf="_300s"
    fig_dir="$OUTPUT_DIR/fig_300s${weighted_suffix}"
  else
    suf="_60s"
    fig_dir="$OUTPUT_DIR/fig_60s${weighted_suffix}"
  fi

  local ls_gat_rel="LS-GAT/results_${gat_id}_year_${seed}${suf}${weighted_suffix}.csv"
  local ls_sgat_rel="LS-SGAT/results_${sgat_id}_year_${seed}${suf}${weighted_suffix}.csv"
  local mixsat_rel="MIXSAT/results_year_${seed}${suf}${weighted_suffix}.csv"
  local mixing_rel="Mixing/results_year_${seed}${suf}${weighted_suffix}.csv"
  local fourier_rel="FourierSAT/results_year_${seed}${suf}${weighted_suffix}.csv"

  # When weighted_suffix is provided, only include FourierSAT and the learned
  # solvers (LS-SGAT and LS-GAT) in the plots. Otherwise include all five.
  local files_list=()
  local names_list=()
  if [ -n "$weighted_suffix" ]; then
    files_list=("$ls_gat_rel" "$ls_sgat_rel" "$fourier_rel")
    names_list=("LS-GAT" "LS-SGAT" "FourierSAT")
  else
    files_list=("$ls_gat_rel" "$ls_sgat_rel" "$mixsat_rel" "$mixing_rel" "$fourier_rel")
    names_list=("LS-GAT" "LS-SGAT" "MixSAT" "Mixing" "FourierSAT")
  fi

  local missing_file=false
  for year in "${years[@]}"; do
    for rel_path in "${files_list[@]}"; do
      local file_path="$OUTPUT_DIR/${rel_path//year/$year}"
      if [ ! -f "$file_path" ]; then
        missing_file=true
        break 2
      fi
    done
  done

  if [ "$missing_file" = true ]; then
    echo "Skipping results_formatter for timeout ${timeout}s due to missing result files."
    return
  fi

  # Build --files and --names arguments dynamically
  python "$PROJECT_ROOT/tools/results_formatter.py" \
    "$OUTPUT_DIR" \
    "$fig_dir" \
    --files "${files_list[@]}" \
    --years "${years[@]}" \
    --names "${names_list[@]}" \
    --best_cost "$best_cost_file" \
    --show_scores \
    --no-latex
}

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
  processed_years_current=()
  seen_weighted=false
  if [ "$timeout" -eq 300 ]; then
    suf="_300s"
  else
    suf="_60s"
  fi

  for year in "${arr[@]}"; do
    directory="$DATA_ROOT/maxsat${year}_sel"

    processed_years+=("$year")
    processed_years_current+=("$year")

    if [ ! -d "$directory" ]; then
      echo "Skipping missing directory: $directory" >&2
      continue
    fi

    # Use find with -print0 to handle filenames with spaces/newlines
    while IFS= read -r -d '' file; do
      # Skip files that do not have 'unw' in their filename
      # if [[ "$file" != *unw* ]]; then
      #   echo "Skipping file $file as it does not contain 'unw' in its name"
      #   continue
      # fi
      # Detect whether the WCNF file is weighted. A simple heuristic is to
      # check for the presence of the 'wcnf' header with weights (e.g., a
      # 'p wcnf' line) and lines where clause weights are not all '1'. We'll
      # conservatively check for any integer weight >1 on clause lines.
      is_weighted=false
      if head -n 200 "$file" | grep -q "^p wcnf"; then
        # Look for clause lines that start with a positive integer weight > 1
        if awk 'BEGIN{w=0} /^[0-9]+ /{ if($1>1){w=1; exit} } END{exit !w}' "$file"; then
          is_weighted=true
        fi
      fi
      weighted_suffix=""
      if [ "$is_weighted" = true ]; then
        weighted_suffix="_weighted"
      fi

      ls_gat_path="$OUTPUT_DIR/LS-GAT/results_${gat_id}_${year}_${seed}${suf}${weighted_suffix}.csv"
      ls_sgat_path="$OUTPUT_DIR/LS-SGAT/results_${sgat_id}_${year}_${seed}${suf}${weighted_suffix}.csv"
      mixsat_path="$OUTPUT_DIR/MIXSAT/results_${year}_${seed}${suf}${weighted_suffix}.csv"
      mixing_path="$OUTPUT_DIR/Mixing/results_${year}_${seed}${suf}${weighted_suffix}.csv"
      fourier_path="$OUTPUT_DIR/FourierSAT/results_${year}_${seed}${suf}${weighted_suffix}.csv"

      # Track whether we created any new result files for this year/timeout
      any_new_results=false

      # LS-GAT (uses GAT model directory and gat_id). If the problem is
      # weighted we still run LS-GAT per user's request.
      if result_exists "$ls_gat_path" "$file"; then
        echo "Skipping LS-GAT (timeout ${timeout}s): result already present for $file"
      else
        if python "$SRC_DIR/solve.py" \
          --solver sgat \
          --train \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --model-dir "$PLOTS_DIR/GAT/" \
          --model-id "$gat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$ls_gat_path"; then
          any_new_results=true
        else
          echo "WARNING: LS-GAT failed for $file (timeout ${timeout}s) -- continuing" >&2
        fi
      fi


      # LS-SGAT (uses SGAT model directory and sgat_id). Run for both weighted
      # and unweighted problems per user's instruction.
      if result_exists "$ls_sgat_path" "$file"; then
        echo "Skipping LS-SGAT (timeout ${timeout}s): result already present for $file"
      else
        if python "$SRC_DIR/solve.py" \
          --solver sgat \
          --train \
          --problem "$file" \
          --timeout "$timeout" \
          --seed "$seed" \
          --model-dir "$PLOTS_DIR/SGAT/Godel" \
          --model-id "$sgat_id" \
          --cuda "$cuda_device" \
          --save-cost-path "$ls_sgat_path"; then
          any_new_results=true
        else
          echo "WARNING: LS-SGAT failed for $file (timeout ${timeout}s) -- continuing" >&2
        fi
      fi

      # For weighted problems: only run FourierSAT and the learned solvers
      # (LS-SGAT, LS-GAT). For unweighted problems: run all solvers.
      if [ "$is_weighted" = true ]; then
        # MIXSAT and Mixing are skipped for weighted instances.
        # FourierSAT
        if result_exists "$fourier_path" "$file"; then
          echo "Skipping FourierSAT (timeout ${timeout}s, weighted): result already present for $file"
        else
          if python "$SRC_DIR/solve.py" \
            --solver fouriersat \
            --problem "$file" \
            --timeout "$timeout" \
            --seed "$seed" \
            --solver-dir "$FOURIER_DIR" \
            --save-cost-path "$fourier_path"; then
            any_new_results=true
          else
            echo "WARNING: FourierSAT failed for $file (timeout ${timeout}s) -- continuing" >&2
          fi
        fi
      else
        # MIXSAT
        if result_exists "$mixsat_path" "$file"; then
          echo "Skipping MIXSAT (timeout ${timeout}s): result already present for $file"
        else
          if python "$SRC_DIR/solve.py" \
            --solver mixsat \
            --problem "$file" \
            --timeout "$timeout" \
            --seed "$seed" \
            --solver-dir "$MIXSAT_DIR" \
            --save-cost-path "$mixsat_path"; then
            any_new_results=true
          else
            echo "WARNING: MIXSAT failed for $file (timeout ${timeout}s) -- continuing" >&2
          fi
        fi

        # Mixing
        if result_exists "$mixing_path" "$file"; then
          echo "Skipping Mixing (timeout ${timeout}s): result already present for $file"
        else
          if python "$SRC_DIR/solve.py" \
            --solver mixing \
            --problem "$file" \
            --timeout "$timeout" \
            --seed "$seed" \
            --solver-dir "$MIXING_DIR" \
            --save-cost-path "$mixing_path"; then
            any_new_results=true
          else
            echo "WARNING: Mixing failed for $file (timeout ${timeout}s) -- continuing" >&2
          fi
        fi

        # FourierSAT
        if result_exists "$fourier_path" "$file"; then
          echo "Skipping FourierSAT (timeout ${timeout}s): result already present for $file"
        else
          if python "$SRC_DIR/solve.py" \
            --solver fouriersat \
            --problem "$file" \
            --timeout "$timeout" \
            --seed "$seed" \
            --solver-dir "$FOURIER_DIR" \
            --save-cost-path "$fourier_path"; then
            any_new_results=true
          else
            echo "WARNING: FourierSAT failed for $file (timeout ${timeout}s) -- continuing" >&2
          fi
        fi
      fi

      if [ "${any_new_results}" = true ]; then
        run_results_formatter "$timeout" "${processed_years_current[@]}"
        # Pass weighted suffix if this file was weighted
        if [ "$is_weighted" = true ]; then
          seen_weighted=true
          run_results_formatter "$timeout" "${processed_years_current[@]}" "_weighted"
        else
          run_results_formatter "$timeout" "${processed_years_current[@]}"
        fi
      else
        echo "No new results for year $year timeout ${timeout}s — skipping results_formatter"
      fi
      # reset flag for next year
      any_new_results=false
    done < <(find "$directory" -type f -name '*.wcnf' -print0)

  done
done

if [ "${#processed_years[@]}" -gt 0 ]; then
  mapfile -t unique_years < <(printf '%s\n' "${processed_years[@]}" | sort -u)

  run_results_formatter 60 "${unique_years[@]}"
  run_results_formatter 300 "${unique_years[@]}"
  # If we encountered any weighted files during the per-timeout loops, run
  # the formatter for aggregated weighted results as well.
  if [ "$seen_weighted" = true ]; then
    run_results_formatter 60 "${unique_years[@]}" "_weighted"
    run_results_formatter 300 "${unique_years[@]}" "_weighted"
  fi
fi
