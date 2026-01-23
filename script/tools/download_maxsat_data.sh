#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
TOOLS_DIR="${PROJECT_ROOT}/tools"
MAXSAT_DIR="${PROJECT_ROOT}/MaxSAT_Dataset"

mkdir -p "${MAXSAT_DIR}"

run_selector() {
  local input_dir="$1"
  local output_dir="$2"
  python3 "${TOOLS_DIR}/selector.py" "${input_dir}" "${output_dir}"
}

clean_directory_contents() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -mindepth 1 -delete
  fi
}

remove_macos_metadata() {
  local dir="$1"
  [[ -d "${dir}" ]] || return
  find "${dir}" -type d -name '__MACOSX' -prune -exec rm -rf {} +
  find "${dir}" -type f -name '._*' -delete
}

download_file() {
  local url="$1"
  local output_path="$2"

  if [[ "${url}" =~ ^https://drive\.google\.com/file/d/([^/]+)/ ]]; then
    download_google_drive "${BASH_REMATCH[1]}" "${output_path}"
    return
  fi

  if [[ "${url}" =~ ^https://drive\.usercontent\.google\.com/download\?id=([^&]+) ]]; then
    download_google_drive "${BASH_REMATCH[1]}" "${output_path}" "${url}"
    return
  fi

  curl --location --fail --show-error --output "${output_path}" "${url}"
}

download_google_drive() {
  local file_id="$1"
  local output_path="$2"
  local provided_url="${3:-}"
  local base_url="https://drive.usercontent.google.com/download?id=${file_id}&confirm=download"
  local download_url="${provided_url:-${base_url}}"

  if command -v wget >/dev/null 2>&1; then
    wget --continue --output-document="${output_path}" "${download_url}"
  else
    curl --location --fail --show-error --output "${output_path}" "${download_url}"
  fi
}

decompress_archives() {
  local target_dir="$1"
  [[ -d "${target_dir}" ]] || return
  if command -v unxz >/dev/null 2>&1; then
    find "${target_dir}" -type f -name '*.xz' -exec unxz -f {} +
  fi
  if command -v gunzip >/dev/null 2>&1; then
    find "${target_dir}" -type f -name '*.gz' -exec gunzip -f {} +
  fi
}

prepare_year() {
  local year="$1"
  shift
  local target_dir="${MAXSAT_DIR}/${year}"
  local output_dir="${MAXSAT_DIR}/${year}_sel"
  local processed_dir="${output_dir}/.processed"

  mkdir -p "${target_dir}"
  mkdir -p "${output_dir}"
  mkdir -p "${processed_dir}"
  pushd "${target_dir}" >/dev/null

  for spec in "$@"; do
    local url filename
    IFS='|' read -r url filename <<<"${spec}"
    local sentinel="${processed_dir}/${filename}.done"

    if [[ -f "${sentinel}" ]]; then
      echo "Skipping ${year} archive ${filename}; already processed." >&2
      continue
    fi

    clean_directory_contents "${target_dir}"
    download_file "${url}" "${filename}"
    unzip -oq "${filename}"
    rm -f "${filename}"

    remove_macos_metadata "${target_dir}"
    decompress_archives "${target_dir}"
    remove_macos_metadata "${target_dir}"
    run_selector "${target_dir}" "${output_dir}"
    clean_directory_contents "${target_dir}"
    touch "${sentinel}"
  done

  popd >/dev/null
  clean_directory_contents "${target_dir}"
}


prepare_year "maxsat2018" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms18_complete_unwt.zip|ms18_complete_unwt.zip" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms18_complete_wt.zip|ms18_complete_wt.zip" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms18_incomplete_unwt.zip|ms18_incomplete_unwt.zip" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms18_incomplete_wt.zip|ms18_incomplete_wt.zip"

prepare_year "maxsat2020" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms20_complete_wt.zip|ms20_complete_wt.zip" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms20_complete_unwt.zip|ms20_complete_unwt.zip" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms20_incomplete_wt.zip|ms20_incomplete_wt.zip" \
  "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms20_incomplete_unwt.zip|ms20_incomplete_unwt.zip"

prepare_year "maxsat2021" \
  "https://www.cs.helsinki.fi/group/coreo/mse2021/mse2021_benchmarks/mse21_complete_wt.zip|mse21_complete_wt.zip" \
  "https://www.cs.helsinki.fi/group/coreo/mse2021/mse2021_benchmarks/mse21_complete_unwt.zip|mse21_complete_unwt.zip" \
  "https://www.cs.helsinki.fi/group/coreo/mse2021/mse2021_benchmarks/mse21_incomplete_unwt.zip|mse21_incomplete_unwt.zip" \
  "https://www.cs.helsinki.fi/group/coreo/mse2021/mse2021_benchmarks/mse21_incomplete_wt.zip|mse21_incomplete_wt.zip"

prepare_year "maxsat2022" \
  "https://drive.google.com/file/d/1j2UcXFId7EDbgaiuaYeUnBqJjlAXo1aK/view?usp=sharing|mse22_complete_wt.zip" \
  "https://drive.google.com/file/d/1ctOvusz0fY0Ju3dwKfZ9zmxpxq_HbYEW/view?usp=sharing|mse22_complete_unwt.zip" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2022-inc-instances/mse22-incomplete-weighted.zip|mse22_incomplete_wt.zip" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2022-inc-instances/mse22-incomplete-unweighted.zip|mse22_incomplete_unwt.zip"

prepare_year "maxsat2023" \
  "https://drive.google.com/file/d/13qDbScs9jU1VaUaq4L7qSGEUrHxC9t6d/view?usp=drive_link|mse23_complete_unwt.zip" \
  "https://drive.google.com/file/d/1pKuQkuTZr7CO3GXmOGRvrMeLTOaw9Fl6/view?usp=drive_link|mse23_complete_wt.zip" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2023-anytime-instances/MSE2023-anytime-W-benchmarks.zip|mse23_anytime_w.zip" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2023-anytime-instances/MSE2023-anytime-UW-benchmarks.zip|mse23_anytime_uw.zip"

prepare_year "maxsat2024" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2024-instances/mse24-exact-unweighted.zip|mse24_exact_unw.zip" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2024-instances/mse24-exact-weighted.zip|mse24_exact_w.zip" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2024-instances/mse24-anytime-unweighted.zip|mse24_anytime_unw.zip" \
  "https://www.cs.helsinki.fi/group/coreo/MSE2024-instances/mse24-anytime-weighted.zip|mse24_anytime_w.zip"

# After preparing years, move specific MS18 benchmark directories into
# maxsat2018_sel/unweighted and maxsat2018_sel/weighted if those
# destination folders do not already exist.
ms18_src_base="${MAXSAT_DIR}/maxsat2018_sel/maxsat_instances/ms_evals/MS18"
ms18_dest_base="${MAXSAT_DIR}/maxsat2018_sel"

if [[ -d "${ms18_src_base}" ]]; then
  # Move unweighted benchmarks if unweighted target doesn't exist yet
  if [[ ! -d "${ms18_dest_base}/unweighted" ]]; then
    echo "Creating ${ms18_dest_base}/unweighted and moving MS18 unweighted benchmarks"
    mkdir -p "${ms18_dest_base}/unweighted"
    if [[ -d "${ms18_src_base}/mse18-complete-unweighted-benchmarks" ]]; then
      mv "${ms18_src_base}/mse18-complete-unweighted-benchmarks" "${ms18_dest_base}/unweighted/"
    fi
    if [[ -d "${ms18_src_base}/mse18-incomplete-unweighted-benchmarks" ]]; then
      mv "${ms18_src_base}/mse18-incomplete-unweighted-benchmarks" "${ms18_dest_base}/unweighted/"
    fi
  fi

  # Move weighted benchmarks if weighted target doesn't exist yet
  if [[ ! -d "${ms18_dest_base}/weighted" ]]; then
    echo "Creating ${ms18_dest_base}/weighted and moving MS18 weighted benchmarks"
    mkdir -p "${ms18_dest_base}/weighted"
    if [[ -d "${ms18_src_base}/mse18-complete-weighted-benchmarks" ]]; then
      mv "${ms18_src_base}/mse18-complete-weighted-benchmarks" "${ms18_dest_base}/weighted/"
    fi
    if [[ -d "${ms18_src_base}/mse18-incomplete-weighted-benchmarks" ]]; then
      mv "${ms18_src_base}/mse18-incomplete-weighted-benchmarks" "${ms18_dest_base}/weighted/"
    fi
  fi
fi

# Generate synthesized MaxSAT instances with G4SATBench SR generator
# Creates maxsat_synth/ under ${MAXSAT_DIR}
python "${PROJECT_ROOT}/../G4SATBench/g4satbench/generators/sr.py" "${MAXSAT_DIR}/maxsat_synth/" \
  --train_instances 1000 \
  --valid_instances 0 \
  --test_instances 0 \
  --min_n 40 \
  --max_n 200


