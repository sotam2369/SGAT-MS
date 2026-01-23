#!/usr/bin/env python3
"""
Compare LM and ModelPredict `.dedup.csv` solver outputs per year, producing
per-year merged CSVs plus aggregate win/loss and satisfaction summaries.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
LM_DIR = ROOT / "solver_output" / "LM"
MODEL_PREDICT_DIR = ROOT / "solver_output" / "ModelPredict"
OUTPUT_DIR = ROOT / "solver_output" / "comparisons"
SUMMARY_TXT = OUTPUT_DIR / "lm_vs_modelpredict_summary.txt"

WCNF_TOP_PATTERN = re.compile(r"^p\s+wcnf\s+\d+\s+\d+(?:\s+(\d+))?", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"_(\d{4})_")


@dataclass
class SolverResult:
    cost: float
    problem: str
    source_file: Path


def find_dedup_files(directory: Path) -> Iterable[Path]:
    return sorted(directory.glob("*.dedup.csv"))


def extract_year_from_filename(path: Path) -> Optional[int]:
    match = YEAR_PATTERN.search(path.stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def load_solver_results(directory: Path) -> Dict[int, Dict[str, SolverResult]]:
    data: Dict[int, Dict[str, SolverResult]] = defaultdict(dict)

    for file_path in find_dedup_files(directory):
        year = extract_year_from_filename(file_path)
        if year is None:
            continue

        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                problem = row["problem"]
                cost_str = row["cost"]
                if not problem:
                    continue
                try:
                    cost = float(cost_str)
                except (TypeError, ValueError):
                    continue
                data[year][problem] = SolverResult(
                    cost=cost,
                    problem=problem,
                    source_file=file_path,
                )

    return data


_wcnf_cache: Dict[str, Tuple[float, int]] = {}


def _parse_wcnf(path: Path) -> Tuple[float, int]:
    total_soft_weight = 0.0
    soft_clause_count = 0
    top_weight: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                match = WCNF_TOP_PATTERN.match(line)
                if match:
                    token = match.group(1)
                    if token is not None:
                        try:
                            top_weight = int(token)
                        except ValueError:
                            top_weight = None
                continue

            parts = line.split()
            if not parts:
                continue
            try:
                weight = int(parts[0])
            except ValueError:
                continue

            if top_weight is not None and weight == top_weight:
                continue

            total_soft_weight += weight
            soft_clause_count += 1

    return total_soft_weight, soft_clause_count


def get_wcnf_stats(path: str) -> Tuple[float, int]:
    if path not in _wcnf_cache:
        _wcnf_cache[path] = _parse_wcnf(Path(path))
    return _wcnf_cache[path]


def compute_satisfaction(total_weight: float, cost: float) -> Optional[float]:
    if total_weight <= 0:
        return None
    return max(0.0, min(1.0, (total_weight - cost) / total_weight))


def compare_year(
    year: int,
    lm_results: Dict[str, SolverResult],
    mp_results: Dict[str, SolverResult],
) -> Dict[str, float]:
    common = sorted(set(lm_results) & set(mp_results))
    if not common:
        raise ValueError(f"No overlapping problems for year {year}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / f"lm_vs_modelpredict_{year}.csv"

    fieldnames = [
        "problem",
        "total_soft_weight",
        "soft_clause_count",
        "cost_LM",
        "cost_ModelPredict",
        "satisfaction_LM",
        "satisfaction_ModelPredict",
        "cost_difference",
        "winner",
        "source_LM",
        "source_ModelPredict",
    ]

    wins = {"LM": 0, "ModelPredict": 0, "tie": 0}
    satisfaction_totals = {"LM": 0.0, "ModelPredict": 0.0}
    satisfaction_counts = {"LM": 0, "ModelPredict": 0}

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for problem in common:
            lm_entry = lm_results[problem]
            mp_entry = mp_results[problem]

            total_weight, soft_count = get_wcnf_stats(problem)
            sat_lm = compute_satisfaction(total_weight, lm_entry.cost)
            sat_mp = compute_satisfaction(total_weight, mp_entry.cost)

            if sat_lm is not None:
                satisfaction_totals["LM"] += sat_lm
                satisfaction_counts["LM"] += 1
            if sat_mp is not None:
                satisfaction_totals["ModelPredict"] += sat_mp
                satisfaction_counts["ModelPredict"] += 1

            if lm_entry.cost < mp_entry.cost:
                winner = "LM"
            elif mp_entry.cost < lm_entry.cost:
                winner = "ModelPredict"
            else:
                winner = "tie"
            wins[winner] += 1

            cost_diff = lm_entry.cost - mp_entry.cost

            writer.writerow(
                {
                    "problem": problem,
                    "total_soft_weight": f"{total_weight:.0f}"
                    if total_weight.is_integer()
                    else f"{total_weight:.6f}",
                    "soft_clause_count": soft_count,
                    "cost_LM": lm_entry.cost,
                    "cost_ModelPredict": mp_entry.cost,
                    "satisfaction_LM": f"{sat_lm:.6f}" if sat_lm is not None else "",
                    "satisfaction_ModelPredict": f"{sat_mp:.6f}"
                    if sat_mp is not None
                    else "",
                    "cost_difference": cost_diff,
                    "winner": winner,
                    "source_LM": lm_entry.source_file.name,
                    "source_ModelPredict": mp_entry.source_file.name,
                }
            )

    total_duels = wins["LM"] + wins["ModelPredict"]
    win_ratio_lm = wins["LM"] / total_duels if total_duels else float("nan")
    win_ratio_mp = wins["ModelPredict"] / total_duels if total_duels else float("nan")
    avg_sat_lm = (
        satisfaction_totals["LM"] / satisfaction_counts["LM"]
        if satisfaction_counts["LM"]
        else float("nan")
    )
    avg_sat_mp = (
        satisfaction_totals["ModelPredict"] / satisfaction_counts["ModelPredict"]
        if satisfaction_counts["ModelPredict"]
        else float("nan")
    )

    return {
        "year": year,
        "problems": len(common),
        "wins_LM": wins["LM"],
        "wins_ModelPredict": wins["ModelPredict"],
        "wins_tie": wins["tie"],
        "win_ratio_LM": win_ratio_lm,
        "win_ratio_ModelPredict": win_ratio_mp,
        "avg_satisfaction_LM": avg_sat_lm,
        "avg_satisfaction_ModelPredict": avg_sat_mp,
        "output_csv": str(output_csv),
    }


def format_float(value: float, precision: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def combine_results_per_year() -> None:
    lm_data = load_solver_results(LM_DIR)
    mp_data = load_solver_results(MODEL_PREDICT_DIR)

    years = sorted(set(lm_data) & set(mp_data))
    if not years:
        raise SystemExit("No overlapping years found between LM and ModelPredict.")

    year_summaries: List[Dict[str, float]] = []
    for year in years:
        summary = compare_year(year, lm_data[year], mp_data[year])
        year_summaries.append(summary)

    with SUMMARY_TXT.open("w", encoding="utf-8") as handle:
        for summary in year_summaries:
            handle.write(f"Year {summary['year']}:\n")
            handle.write(f"  Problems compared : {summary['problems']}\n")
            handle.write(
                f"  Win counts        : LM {summary['wins_LM']}, "
                f"ModelPredict {summary['wins_ModelPredict']}, "
                f"Ties {summary['wins_tie']}\n"
            )
            handle.write(
                "  Win ratios        : LM "
                f"{format_float(summary['win_ratio_LM'], 3)}, "
                f"ModelPredict {format_float(summary['win_ratio_ModelPredict'], 3)}\n"
            )
            handle.write(
                "  Avg satisfaction  : LM "
                f"{format_float(summary['avg_satisfaction_LM'])}, "
                "ModelPredict "
                f"{format_float(summary['avg_satisfaction_ModelPredict'])}\n"
            )
            handle.write(f"  Output CSV        : {summary['output_csv']}\n\n")

    print("Per-year comparison completed.")
    print(f"  Summary file : {SUMMARY_TXT}")
    for summary in year_summaries:
        print(
            f"  Year {summary['year']}: wins LM {summary['wins_LM']} / "
            f"ModelPredict {summary['wins_ModelPredict']} / ties {summary['wins_tie']}, "
            f"avg satisfaction LM {format_float(summary['avg_satisfaction_LM'])}, "
            f"ModelPredict {format_float(summary['avg_satisfaction_ModelPredict'])}"
        )


if __name__ == "__main__":
    combine_results_per_year()
