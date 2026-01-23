#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
TOOLS_DIR="$PROJECT_ROOT/tools"
PLOTS_DIR="$PROJECT_ROOT/plots"
DATASET_DIR="$PROJECT_ROOT/dataset"

dtrain="$DATASET_DIR/ms2018_train_2.0-0_nfs_nsl.pt"
dtest="$DATASET_DIR/ms2018_test_2.0-0_nfs_nsl.pt"

# Helper: run a command but don't let non-zero exit stop the whole script.
# Uses "if ! cmd; then ... fi" so `set -e` doesn't abort the script when the command fails.
run_allow_fail() {
  echo "+ Running: $*"
  if ! "$@"; then
    rc=$?
    echo "WARNING: command failed with exit code $rc -- continuing..." >&2
    return $rc
  fi
}

# Experiment for SGATs
run_allow_fail python "$SRC_DIR/main.py" \
  --epochs 2500 \
  -ly 6 \
  --heads 2 \
  --hidden 4 \
  --t-norm godel \
  -dtrain "$dtrain" \
  -dtest "$dtest" \
  -trs 8 \
  -tes 2 \
  -b 4 \
  -oe 10 \
  -c cuda:0 \
  --optimizer Adam \
  -fr 5000 \
  --best-weights 1 \
  --trans-prob 1 \
  --dir "$PLOTS_DIR/SGAT/Godel" \
  -lr 0.002 \
  -ed 4 \
  --normalization


# Experiment for Traditional GATs
run_allow_fail python "$SRC_DIR/main.py" \
  --epochs 2500 \
  -ly 6 \
  --heads 2 \
  --hidden 4 \
  -dtrain "$dtrain" \
  -dtest "$dtest" \
  -trs 8 \
  -tes 2 \
  -b 4 \
  -oe 10 \
  -c cuda:0 \
  --optimizer Adam \
  -fr 5000 \
  --best-weights 1 \
  --trans-prob 1 \
  --dir "$PLOTS_DIR/GAT/" \
  -lr 0.002 \
  -ed 4 \
  --normalization \
  --use-gat

# Visualization
remove_plots=50

run_allow_fail python "$TOOLS_DIR/plot_results_combiner.py" \
    --ids 1 1 \
    --names GAT SGAT \
    --input_dir "$PLOTS_DIR/GAT" "$PLOTS_DIR/SGAT/Godel" \
    --file eval \
    --use-min \
    --remove_plots "$remove_plots"
