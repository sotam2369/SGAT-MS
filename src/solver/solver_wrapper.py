import abc
import os
import subprocess
import sys
from typing import Optional


class Solver(abc.ABC):
    """Abstract base class for solver wrappers."""

    @abc.abstractmethod
    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        """Run the solver and return the best cost, or -1 on failure."""
        raise NotImplementedError


def _parse_optimal_from_lines(lines: list[str], keyword: str = "o", pos: int = 1) -> int:
    """Utility to parse the optimal value from solver output lines."""
    for line in lines:
        line_split = line.split()
        if len(line_split) > pos and line_split[0] == keyword:
            try:
                return int(line_split[pos])
            except Exception:
                continue
    return -1


def _abs_path(path: Optional[str]) -> str:
    return os.path.abspath(path or ".")


def _normalize_problem_path(problem_file: str) -> str:
    return os.path.abspath(problem_file)


def _write_init_file(solver_dir: str, filename: str, init_values: Optional[list[float]]) -> None:
    if init_values is None:
        return
    os.makedirs(solver_dir, exist_ok=True)
    file_path = os.path.join(solver_dir, filename)
    with open(file_path, "w", encoding="utf-8") as handle:
        for value in init_values:
            handle.write(f"{float(value)}\n")


def _run_solver_command(cmd: list[str], timeout: int, solver_dir: str) -> list[str]:
    """Run a solver command and return stdout split into lines."""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            cwd=solver_dir,
            check=False,
        )
        stdout = completed.stdout
    except subprocess.TimeoutExpired as err:
        stdout = err.output or b""
    except Exception:
        return []

    try:
        return stdout.decode().splitlines()
    except Exception:
        return []


class NuWLSSolver(Solver):
    """Wrapper for the NuWLS solver."""

    INIT_FILENAME = "prediction_file.csv"

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        solver_dir_abs = _abs_path(solver_dir)
        problem_path = _normalize_problem_path(problem_file)
        _write_init_file(solver_dir_abs, self.INIT_FILENAME, init_values)
        cmd = [
            "./starexec_nuwls-we-with-runsolver.sh",
            problem_path,
            "1",
            str(timeout),
        ]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        if not output_lines:
            return -1
        for line in output_lines:
            tokens = line.split()
            if len(tokens) > 2 and tokens[1] == "o":
                try:
                    return int(tokens[2])
                except Exception:
                    continue
        return -1


class MixingSolver(Solver):
    """Wrapper for the Mixing solver."""

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        solver_dir_abs = _abs_path(solver_dir)
        problem_path = _normalize_problem_path(problem_file)
        cmd = ["./mixing", problem_path]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        if not output_lines:
            return -1

        for line in reversed(output_lines):
            parts = line.split()
            if len(parts) >= 8:
                try:
                    satisfied, total = map(int, parts[7].split("/"))
                    return total - satisfied
                except Exception:
                    continue
        return -1


class MixSATSolver(Solver):
    """Wrapper for the MixSAT solver."""

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        solver_dir_abs = _abs_path(solver_dir)
        problem_path = _normalize_problem_path(problem_file)
        cmd = ["./incomplete", problem_path]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        if not output_lines:
            return -1

        for line in reversed(output_lines):
            if "best" in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        return int(parts[1])
                    except Exception:
                        continue
        return -1


class BandHSSolver(Solver):
    """Wrapper for the BandHS solver."""

    INIT_FILENAME = "prediction_file.csv"

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        solver_dir_abs = _abs_path(solver_dir)
        problem_path = _normalize_problem_path(problem_file)
        _write_init_file(solver_dir_abs, self.INIT_FILENAME, init_values)
        cmd = ["./BandHS", problem_path]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        if not output_lines:
            return -1
        return _parse_optimal_from_lines(output_lines, keyword="o", pos=1)


class SATLikeSolver(Solver):
    """Wrapper for the SATLike3.0 solver."""

    INIT_FILENAME = "prediction_file.csv"

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        solver_dir_abs = _abs_path(solver_dir)
        problem_path = _normalize_problem_path(problem_file)
        _write_init_file(solver_dir_abs, self.INIT_FILENAME, init_values)
        cmd = ["./SATLike3.0", problem_path]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        if not output_lines:
            return -1
        return _parse_optimal_from_lines(output_lines, keyword="o", pos=1)


class FourierSATSolver(Solver):
    """Wrapper for the FourierSAT solver."""

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        solver_dir_abs = _abs_path(solver_dir)
        problem_path = _normalize_problem_path(problem_file)
        cmd = [
            sys.executable,
            "FourierSAT.py",
            problem_path,
            "--ismaxsat",
            "1",
            "--timelimit",
            str(timeout),
        ]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        if not output_lines:
            return -1
        return _parse_optimal_from_lines(output_lines, keyword="o", pos=1)


class SPBSolver(Solver):
    """Wrapper for the SPB-MaxSAT solver."""

    INIT_FILENAME = "prediction_file.csv"

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        solver_dir_abs = _abs_path(solver_dir)
        problem_path = _normalize_problem_path(problem_file)
        _write_init_file(solver_dir_abs, self.INIT_FILENAME, init_values)
        cmd = ["./SPB-MaxSAT", problem_path]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        if not output_lines:
            return -1
        return _parse_optimal_from_lines(output_lines, keyword="o", pos=1)
