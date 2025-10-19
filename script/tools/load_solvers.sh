#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
SOLVER_ARCHIVE_DIR="${PROJECT_ROOT}/solvers"

extract_solver_archive() {
  local archive_name="$1"
  local archive_path="${SOLVER_ARCHIVE_DIR}/${archive_name}"
  if [[ ! -f "${archive_path}" ]]; then
    echo "Missing solver archive: ${archive_path}" >&2
    exit 1
  fi
  unzip -q -o "${archive_path}" -d "${PROJECT_ROOT}"
}

extract_solver_archive "NuWLS.zip"
extract_solver_archive "BandHS.zip"
extract_solver_archive "SATLike3.0.zip"
extract_solver_archive "SPB-MaxSAT.zip"

if [[ ! -d "${PROJECT_ROOT}/mixing" ]]; then
  git clone https://github.com/locuslab/mixing.git "${PROJECT_ROOT}/mixing"
fi
(
  cd "${PROJECT_ROOT}/mixing"
  make
)

if [[ ! -d "${PROJECT_ROOT}/mixsat" ]]; then
  git clone https://github.com/locuslab/mixsat.git "${PROJECT_ROOT}/mixsat"
fi
(
  cd "${PROJECT_ROOT}/mixsat"
  make
)
