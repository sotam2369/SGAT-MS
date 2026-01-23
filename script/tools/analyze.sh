#!/usr/bin/env bash
set -euo pipefail

# Analyze MaxSAT datasets for unique weighted/unweighted instance counts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/MaxSAT_Dataset"

python3 - <<'PY' "${DATA_ROOT}"
import os
import re
import sys
import hashlib
from collections import defaultdict

DATA_ROOT = sys.argv[1]
YEARS = range(2020, 2025)
YEARS = [2021]
BAR_WIDTH = 32
NSOFT_WTS_RE = re.compile(r'"nsoft_wts"\s*:\s*(\d+)', re.IGNORECASE)


def list_instances(year_dir):
    for dirpath, _, filenames in os.walk(year_dir):
        for name in sorted(filenames):
            if name.lower().endswith(".wcnf"):
                rel_path = os.path.relpath(os.path.join(dirpath, name), year_dir)
                yield rel_path.replace(os.sep, "/"), name


def render_progress(current, total):
    if total == 0:
        return
    ratio = current / total
    filled = int(ratio * BAR_WIDTH)
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    print(f"\r    [{bar}] {current:>4}/{total:<4} ({ratio * 100:5.1f}%)", end="", flush=True)


def classify_instance(file_path):
    top_weight = None
    saw_soft_clause = False
    has_weight_gt_one = False
    hint_category = None
    hasher = hashlib.sha1()

    try:
        with open(file_path, "rb") as handle:
            for raw_line in handle:
                hasher.update(raw_line)
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("c"):
                    match = NSOFT_WTS_RE.search(line)
                    if match:
                        value = int(match.group(1))
                        hint_category = "weighted" if value > 1 else "unweighted"
                    continue
                if line.startswith("p "):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            top_weight = int(parts[3])
                        except ValueError:
                            top_weight = None
                    continue

                parts = line.split()
                try:
                    weight = int(parts[0])
                except ValueError:
                    continue

                # Treat clauses with weight equal to top weight as hard clauses.
                if top_weight is not None and weight == top_weight:
                    continue

                saw_soft_clause = True
                if weight != 1:
                    has_weight_gt_one = True
        digest = hasher.hexdigest()
    except OSError as exc:
        rel = os.path.relpath(file_path, DATA_ROOT)
        print(f"    Error reading {rel}: {exc}")
        return "unknown", False, None

    if has_weight_gt_one:
        return "weighted", True, digest
    if hint_category:
        return hint_category, False, digest
    if saw_soft_clause:
        return "unweighted", True, digest
    return "unknown", False, digest


summary = []
missing_years = []

for year in YEARS:
    year_dir = os.path.join(DATA_ROOT, f"maxsat{year}_sel")
    if not os.path.isdir(year_dir):
        missing_years.append(year)
        continue

    instances = list(list_instances(year_dir))
    total_files = len(instances)
    print(f"Year {year}: scanning {total_files} files")

    category_stats = {
        "weighted": {
            "paths": [],
            "hash_to_paths": defaultdict(list),
            "inferred": 0,
        },
        "unweighted": {
            "paths": [],
            "hash_to_paths": defaultdict(list),
            "inferred": 0,
        },
        "unknown": {
            "paths": [],
            "hash_to_paths": defaultdict(list),
            "inferred": 0,
        },
    }

    for index, (rel_path, name) in enumerate(instances, start=1):
        full_path = os.path.join(year_dir, rel_path.replace("/", os.sep))
        category, inferred, digest = classify_instance(full_path)
        if digest is None:
            render_progress(index, total_files)
            continue

        stats = category_stats.setdefault(
            category,
            {"paths": [], "hash_to_paths": defaultdict(list), "inferred": 0},
        )
        stats["paths"].append(rel_path)
        stats["hash_to_paths"][digest].append(rel_path)
        if inferred:
            stats["inferred"] += 1

        render_progress(index, total_files)

    if total_files:
        print()

    weighted_data = category_stats.get("weighted", {})
    unweighted_data = category_stats.get("unweighted", {})
    unknown_data = category_stats.get("unknown", {})

    def summarize(data):
        if not data:
            return 0, 0, []
        unique = len(data["hash_to_paths"])
        duplicates = sum(len(paths) - 1 for paths in data["hash_to_paths"].values())
        duplicate_examples = [
            paths for paths in data["hash_to_paths"].values() if len(paths) > 1
        ]
        return unique, duplicates, duplicate_examples

    weighted_unique, weighted_duplicates, weighted_dupe_paths = summarize(weighted_data)
    unweighted_unique, unweighted_duplicates, unweighted_dupe_paths = summarize(
        unweighted_data
    )
    unknown_unique, _, _ = summarize(unknown_data)

    weighted_total = len(weighted_data.get("paths", []))
    unweighted_total = len(unweighted_data.get("paths", []))
    unknown_total = len(unknown_data.get("paths", []))

    if weighted_total:
        print(f"    Weighted files scanned:  {weighted_total}")
    print(f"    Weighted unique instances:   {weighted_unique}")
    if weighted_duplicates:
        print(f"    Weighted duplicates skipped: {weighted_duplicates}")
        for dup in weighted_dupe_paths[:3]:
            others = ", ".join(dup[1:2])
            suffix = "" if len(dup) <= 2 else f" (and {len(dup) - 2} more)"
            if others:
                print(f"        - duplicates: {dup[0]}, {others}{suffix}")
            else:
                print(f"        - duplicates: {dup[0]}{suffix}")
    if unweighted_total:
        print(f"    Unweighted files scanned: {unweighted_total}")
    print(f"    Unweighted unique instances: {unweighted_unique}")
    if unweighted_duplicates:
        print(f"    Unweighted duplicates skipped: {unweighted_duplicates}")
        for dup in unweighted_dupe_paths[:3]:
            others = ", ".join(dup[1:2])
            suffix = "" if len(dup) <= 2 else f" (and {len(dup) - 2} more)"
            if others:
                print(f"        - duplicates: {dup[0]}, {others}{suffix}")
            else:
                print(f"        - duplicates: {dup[0]}{suffix}")
    if unweighted_data.get("inferred"):
        print(
            f"    Assumed unweighted after scanning clauses: {unweighted_data['inferred']}"
        )
    if unknown_unique:
        print(
            f"    Warning: {unknown_unique} files could not be classified (from {unknown_total} scanned)"
        )
        example = next(iter(unknown_data["hash_to_paths"].values()))
        print(f"        e.g. {example[0]}")

    weighted_hashes = set(weighted_data.get("hash_to_paths", {}))
    unweighted_hashes = set(unweighted_data.get("hash_to_paths", {}))
    overlap_hashes = weighted_hashes & unweighted_hashes
    if overlap_hashes:
        print(
            f"    Note: {len(overlap_hashes)} file contents appear in both weighted and unweighted groups"
        )
        sample_hash = next(iter(overlap_hashes))
        weighted_example = weighted_data["hash_to_paths"][sample_hash][0]
        unweighted_example = unweighted_data["hash_to_paths"][sample_hash][0]
        print(f"        weighted:   {weighted_example}")
        print(f"        unweighted: {unweighted_example}")

    summary.append((year, weighted_unique, unweighted_unique))

if missing_years:
    print("Missing datasets:")
    for year in missing_years:
        print(f"  - maxsat{year}_sel")

if summary:
    print("\nSummary:")
    header = ("Year", "Weighted", "Unweighted")
    rows = [header] + [(str(year), str(weighted), str(unweighted)) for year, weighted, unweighted in summary]
    col_widths = [max(len(row[col]) for row in rows) for col in range(3)]
    for row in rows:
        print("  " + "  ".join(row[col].ljust(col_widths[col]) for col in range(3)))

print("\nList of files:")
print("Weighted files:")
for path in weighted_data.get("paths", []):
    print(f"  - {path}")

print("Unweighted files:")
for path in unweighted_data.get("paths", []):
    print(f"  - {path}")
PY
