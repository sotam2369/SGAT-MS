#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
SOLVER_ARCHIVE_DIR="${PROJECT_ROOT}/solvers"
EXTRACTION_DIR="$(cd -- "${PROJECT_ROOT}/.." >/dev/null 2>&1 && pwd)"

mkdir -p "${EXTRACTION_DIR}"

extract_solver_archive() {
  local archive_name="$1"
  local archive_path="${SOLVER_ARCHIVE_DIR}/${archive_name}"
  if [[ ! -f "${archive_path}" ]]; then
    echo "Missing solver archive: ${archive_path}" >&2
    exit 1
  fi
  unzip -q -o "${archive_path}" -d "${EXTRACTION_DIR}"
}

extract_solver_archive "NuWLS.zip"
extract_solver_archive "BandHS.zip"
extract_solver_archive "SATLike3.0.zip"
extract_solver_archive "SPB-MaxSAT.zip"
extract_solver_archive "FourierSAT.zip"

if [[ ! -d "${EXTRACTION_DIR}/mixing" ]]; then
  git clone https://github.com/locuslab/mixing.git "${EXTRACTION_DIR}/mixing"
fi
(
  cd "${EXTRACTION_DIR}/mixing"
  make
)

if [[ ! -d "${EXTRACTION_DIR}/mixsat" ]]; then
  git clone https://github.com/locuslab/mixsat.git "${EXTRACTION_DIR}/mixsat"
fi
(
  cd "${EXTRACTION_DIR}/mixsat"
  make
)

if [[ ! -d "${EXTRACTION_DIR}/G4SATBench" ]]; then
  git clone https://github.com/sotam2369/G4SATBench.git "${EXTRACTION_DIR}/G4SATBench"
fi