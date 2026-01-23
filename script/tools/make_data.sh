#!/usr/bin/env bash
set -euo pipefail

# Resolve project root so the script works from any current directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
UTILS_DIR="${PROJECT_ROOT}/src/utils"
DATASET_DIR="${PROJECT_ROOT}/dataset"
MAXSAT_SOURCE_DIR="${PROJECT_ROOT}/MaxSAT_Dataset/maxsat2018_sel"
OUTPUT_FILE="${DATASET_DIR}/ms2018_type_maxsize-minsize_nfs_nsl.pt"

mkdir -p "${DATASET_DIR}"

# python3 "${UTILS_DIR}/dataset_loader.py" \
#   -dir "${MAXSAT_SOURCE_DIR}" \
#   -out "${OUTPUT_FILE}" \
#   --split 10 \
#   -ms 2 \
#   --tsplit 0.2 \
#   --ftype 0


MAXSAT_SOURCE_DIR="${PROJECT_ROOT}/MaxSAT_Dataset/maxsat2018_sel/unweighted/"
OUTPUT_FILE="${DATASET_DIR}/ms2018_uw_type_maxsize-minsize_nfs_nsl.pt"
G4SB_OUT="${PROJECT_ROOT}/MaxSAT_Dataset/maxsat2018_g4satbench"
rm -rf "${G4SB_OUT}"
mkdir -p "${G4SB_OUT}/train" "${G4SB_OUT}/valid"
python3 "${UTILS_DIR}/dataset_loader.py" \
  -dir "${MAXSAT_SOURCE_DIR}" \
  -out "${OUTPUT_FILE}" \
  --split 10 \
  -ms 2 \
  --tsplit 0.2 \
  --ftype 0 \
  --copy-out "${G4SB_OUT}"

MAXSAT_SOURCE_DIR="${PROJECT_ROOT}/MaxSAT_Dataset/maxsat_synth/train/unsat/"
OUTPUT_FILE="${DATASET_DIR}/ms_synth_type_maxsize-minsize_nfs_nsl.pt"
G4SB_OUT="${PROJECT_ROOT}/MaxSAT_Dataset/maxsat_synth_g4satbench"
rm -rf "${G4SB_OUT}"
mkdir -p "${G4SB_OUT}/train" "${G4SB_OUT}/valid"
python3 "${UTILS_DIR}/dataset_loader.py" \
  -dir "${MAXSAT_SOURCE_DIR}" \
  -out "${OUTPUT_FILE}" \
  --split 10 \
  -ms 2 \
  --tsplit 0.2 \
  --ftype 0 \
  --copy-out "${G4SB_OUT}"
