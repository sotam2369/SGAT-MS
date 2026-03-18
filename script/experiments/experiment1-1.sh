#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
TOOLS_DIR="$PROJECT_ROOT/tools"
PLOTS_DIR="$PROJECT_ROOT/plots"
DATASET_DIR="$PROJECT_ROOT/dataset"

dtrain="$DATASET_DIR/ms_synth_train_2.0-0_nfs_nsl.pt"
dtest="$DATASET_DIR/ms2018_uw_test_2.0-0_nfs_nsl.pt"

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

# Experiment for SGATs: run the block with multiple seeds
for seed in 1 2 3; do
  echo "=== Running experiments with seed=$seed ==="

  # per-seed plots dir
  mkdir -p "$PLOTS_DIR/SGAT_uw_synth/Godel" "$PLOTS_DIR/G4SATBench/GGNN_synth" "$PLOTS_DIR/G4SATBench/MS-ESFG_synth" "$PLOTS_DIR/G4SATBench/NeuroSAT_synth" || true

  python "$PROJECT_ROOT/../G4SATBench/train_model.py" assignment \
    "$PROJECT_ROOT/MaxSAT_Dataset/maxsat_synth_g4satbench/train" \
    --out_dir "$PLOTS_DIR/G4SATBench/MS-ESFG_synth" \
    --train_splits unsat \
    --valid_dir "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/valid" \
    --valid_splits unsat \
    --loss unsupervised_2 \
    --graph vcg \
    --model ms_esfg \
    --n_iterations 20 \
    --dim 128 \
    --lr 2e-5 \
    --weight_decay 1e-10 \
    --scheduler ReduceLROnPlateau \
    --batch_size 16 \
    --seed "$seed" \
    --epochs 100

  run_allow_fail python "$SRC_DIR/main.py" \
    --epochs 100 \
    -ly 6 \
    --heads 2 \
    --hidden 4 \
    --t-norm godel \
    -dtrain "$dtrain" \
    -dtest "$dtest" \
    -trs 8 \
    -tes 2 \
    -b 16 \
    -oe 10 \
    -c cuda:0 \
    --optimizer NAdam \
    -fr 5000 \
    --best-weights 1 \
    --trans-prob 1 \
    --dir "$PLOTS_DIR/SGAT_uw_synth/Godel" \
    -lr 0.002 \
    -ed 4 \
    --normalization \
    --seed "$seed"

  python "$PROJECT_ROOT/../G4SATBench/train_model.py" assignment \
    "$PROJECT_ROOT/MaxSAT_Dataset/maxsat_synth_g4satbench/train" \
    --out_dir "$PLOTS_DIR/G4SATBench/GGNN_synth" \
    --train_splits unsat \
    --valid_dir "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/valid" \
    --valid_splits unsat \
    --loss unsupervised_2 \
    --graph vcg \
    --model ggnn \
    --n_iterations 32 \
    --lr 0.002 \
    --weight_decay 1e-08 \
    --scheduler ReduceLROnPlateau \
    --batch_size 16 \
    --seed "$seed" \
    --epochs 100

  python "$PROJECT_ROOT/../G4SATBench/train_model.py" assignment \
    "$PROJECT_ROOT/MaxSAT_Dataset/maxsat_synth_g4satbench/train" \
    --out_dir "$PLOTS_DIR/G4SATBench/NeuroSAT_synth" \
    --train_splits unsat \
    --valid_dir "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/valid" \
    --valid_splits unsat \
    --loss unsupervised_2 \
    --graph lcg \
    --model neurosat \
    --n_iterations 32 \
    --lr 0.002 \
    --weight_decay 1e-08 \
    --scheduler ReduceLROnPlateau \
    --batch_size 16 \
    --seed "$seed" \
    --epochs 100

  echo "=== Finished seed=$seed ==="
done




dtrain="$DATASET_DIR/ms2018_uw_train_2.0-0_nfs_nsl.pt"
dtest="$DATASET_DIR/ms2018_uw_test_2.0-0_nfs_nsl.pt"

# Experiment for SGATs: run the block with multiple seeds
for seed in 1 2 3; do
  echo "=== Running experiments with seed=$seed ==="

  # per-seed plots dir
  mkdir -p "$PLOTS_DIR/SGAT_uw/Godel" "$PLOTS_DIR/G4SATBench/GGNN" "$PLOTS_DIR/G4SATBench/MS-ESFG" "$PLOTS_DIR/G4SATBench/NeuroSAT" || true

  run_allow_fail python "$SRC_DIR/main.py" \
    --epochs 100 \
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
    --optimizer NAdam \
    -fr 5000 \
    --best-weights 1 \
    --trans-prob 1 \
    --dir "$PLOTS_DIR/SGAT_uw/Godel" \
    -lr 0.002 \
    -ed 4 \
    --normalization \
    --seed "$seed"

  python "$PROJECT_ROOT/../G4SATBench/train_model.py" assignment \
    "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/train" \
    --out_dir "$PLOTS_DIR/G4SATBench/GGNN" \
    --train_splits unsat \
    --valid_dir "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/valid" \
    --valid_splits unsat \
    --loss unsupervised_2 \
    --graph vcg \
    --model ggnn \
    --n_iterations 32 \
    --lr 0.002 \
    --weight_decay 1e-08 \
    --scheduler ReduceLROnPlateau \
    --batch_size 1 \
    --seed "$seed" \
    --epochs 100

  python "$PROJECT_ROOT/../G4SATBench/train_model.py" assignment \
    "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/train" \
    --out_dir "$PLOTS_DIR/G4SATBench/MS-ESFG" \
    --train_splits unsat \
    --valid_dir "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/valid" \
    --valid_splits unsat \
    --loss unsupervised_2 \
    --graph vcg \
    --model ms_esfg \
    --n_iterations 20 \
    --dim 128 \
    --lr 2e-5 \
    --weight_decay 1e-10 \
    --scheduler ReduceLROnPlateau \
    --batch_size 1 \
    --seed "$seed" \
    --epochs 100

  python "$PROJECT_ROOT/../G4SATBench/train_model.py" assignment \
    "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/train" \
    --out_dir "$PLOTS_DIR/G4SATBench/NeuroSAT" \
    --train_splits unsat \
    --valid_dir "$PROJECT_ROOT/MaxSAT_Dataset/maxsat2018_g4satbench/valid" \
    --valid_splits unsat \
    --loss unsupervised_2 \
    --graph lcg \
    --model neurosat \
    --n_iterations 32 \
    --lr 0.002 \
    --weight_decay 1e-08 \
    --scheduler ReduceLROnPlateau \
    --batch_size 1 \
    --seed "$seed" \
    --epochs 100

  echo "=== Finished seed=$seed ==="
done

# Visualization

remove_plots=10

# Visualization for synth models (all in one call)
run_allow_fail python "$TOOLS_DIR/plot_results_combiner.py" \
  --ids "2,3,4" "2,3,4" "1,2,3" "1,2,3" \
  --names SGAT GGNN MS-ESFG NeuroSAT \
  --input_dir "$PLOTS_DIR/SGAT_uw_synth/Godel" "$PLOTS_DIR/G4SATBench/GGNN_synth" "$PLOTS_DIR/G4SATBench/MS-ESFG_synth" "$PLOTS_DIR/G4SATBench/NeuroSAT_synth" \
  --file eval \
  --use-min \
  --remove_plots "$remove_plots" \
  --ymin 0.3 --ymax 1 \
  --figsize 7 6 \
  --save_name "eval_architecture_synth"

# Visualization for normal models (all in one call)
run_allow_fail python "$TOOLS_DIR/plot_results_combiner.py" \
  --ids "1,2,3" "2,3,4" "1,2,3" "1,2,3" \
  --names SGAT GGNN MS-ESFG NeuroSAT \
  --input_dir "$PLOTS_DIR/SGAT_uw/Godel" "$PLOTS_DIR/G4SATBench/GGNN" "$PLOTS_DIR/G4SATBench/MS-ESFG" "$PLOTS_DIR/G4SATBench/NeuroSAT" \
  --file eval \
  --use-min \
  --remove_plots "$remove_plots" \
  --ymin 0.3 --ymax 1 \
  --figsize 7 6 \
  --save_name "eval_architecture_normal"
