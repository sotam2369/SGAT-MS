import abc
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Tuple


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


@dataclass
class _WeightedFormula:
    nv: int
    hard: List[List[int]]
    soft: List[List[int]]
    wght: List[float]


def _parse_clause_literals(tokens: List[str]) -> List[int]:
    """Parse a DIMACS clause line into a list of literals."""
    literals: List[int] = []
    for token in tokens:
        if token == "0":
            break
        try:
            literals.append(int(token))
        except ValueError:
            continue
    return literals


def _load_weighted_formula(problem_path: str) -> _WeightedFormula:
    """Load a CNF or WCNF file without using PySAT."""
    hard: List[List[int]] = []
    soft: List[List[int]] = []
    soft_weights: List[float] = []
    num_vars = 0
    top_weight: Optional[float] = None
    format_hint = None

    try:
        format_hint = problem_path.lower().endswith(".wcnf")
    except Exception:
        format_hint = None

    with open(problem_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0].lower() == "p" and len(parts) >= 3:
                # Header: p cnf <n_vars> <n_clauses> [top]
                format_token = parts[1].lower()
                if format_token == "wcnf":
                    format_hint = True
                    if len(parts) >= 5:
                        try:
                            top_weight = float(parts[4])
                        except ValueError:
                            top_weight = None
                else:
                    format_hint = False
                try:
                    num_vars = int(parts[2])
                except ValueError:
                    num_vars = 0
                continue

            if format_hint:
                try:
                    weight = float(parts[0])
                except ValueError:
                    continue
                literals = _parse_clause_literals(parts[1:])
                if not literals:
                    continue
                if top_weight is not None and weight >= top_weight:
                    hard.append(literals)
                else:
                    soft.append(literals)
                    soft_weights.append(weight)
            else:
                literals = _parse_clause_literals(parts)
                if literals:
                    hard.append(literals)

    if num_vars == 0:
        num_vars = max((abs(lit) for clause in hard + soft for lit in clause), default=0)

    if format_hint and top_weight is not None:
        # Ensure any clause with weight >= top is treated as hard even if it slipped into soft.
        promoted_clauses = []
        promoted_weights = []
        for clause, weight in zip(list(soft), list(soft_weights)):
            if weight >= top_weight:
                hard.append(clause)
            else:
                promoted_clauses.append(clause)
                promoted_weights.append(weight)
        soft = promoted_clauses
        soft_weights = promoted_weights

    return _WeightedFormula(nv=num_vars, hard=hard, soft=soft, wght=soft_weights)


def _evaluate_clauses(clauses: List[List[int]], assignment: List[int]) -> Tuple[int, float]:
    """Return (unsatisfied_count, accuracy) for the given assignment."""
    total = len(clauses)
    if total == 0:
        return 0, 1.0

    unsatisfied = 0
    for clause in clauses:
        clause_satisfied = any(
            (lit > 0 and assignment[abs(lit) - 1] == 1)
            or (lit < 0 and assignment[abs(lit) - 1] == 0)
            for lit in clause
        )
        if not clause_satisfied:
            unsatisfied += 1

    accuracy = (total - unsatisfied) / total
    return unsatisfied, accuracy


def _convert_wcnf_to_cnf(wcnf_path: str, out_dir: str) -> str:
    """Conservatively convert a WCNF file to plain CNF by stripping weights.

    This function treats every clause in the WCNF (hard and soft) as an
    unweighted clause in the output CNF. The number of variables is taken from
    the 'p wcnf' header when present; otherwise it is inferred from the
    largest variable index seen.

    The converted file is written into out_dir with suffix '_converted.cnf'.
    Returns the absolute path to the converted CNF file.
    """
    os.makedirs(out_dir or '.', exist_ok=True)
    basename = os.path.splitext(os.path.basename(wcnf_path))[0]
    out_path = os.path.join(out_dir, f"{basename}_converted.cnf")

    max_var = 0
    clauses = []
    num_vars_from_header = None

    with open(wcnf_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p' and len(parts) >= 3:
                # p wcnf <num_vars> <num_clauses> [top]
                try:
                    if parts[1].lower() == 'wcnf' and len(parts) >= 3:
                        num_vars_from_header = int(parts[2])
                except Exception:
                    pass
                continue

            # Clause line: <weight> lit1 lit2 ... 0
            # We ignore the weight and only keep the literals up to the terminating 0.
            try:
                # find '0' terminator
                if '0' in parts:
                    zero_idx = parts.index('0')
                    lits = parts[1:zero_idx]
                else:
                    # malformed but try: take all except first token (weight)
                    lits = parts[1:]
                if not lits:
                    continue
                # update max variable index
                for lit in lits:
                    try:
                        v = abs(int(lit))
                        if v > max_var:
                            max_var = v
                    except Exception:
                        continue
                clauses.append(' '.join(lits) + ' 0')
            except Exception:
                continue

    num_vars = num_vars_from_header or max_var
    num_clauses = len(clauses)

    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(f"p cnf {num_vars} {num_clauses}\n")
        for c in clauses:
            out.write(c + '\n')

    return os.path.abspath(out_path)


def _ensure_cnf(problem_file: str, solver_dir: str) -> str:
    """Return a path to a CNF file usable by solvers.

    If the input is already a CNF (header 'p cnf' or .cnf extension) returns
    the original path. If it is a WCNF (header 'p wcnf' or .wcnf extension)
    it is converted to CNF using _convert_wcnf_to_cnf and the converted path is
    returned.
    """
    # Fast path: extension
    lower = problem_file.lower()
    try:
        with open(problem_file, 'r', encoding='utf-8', errors='ignore') as f:
            # inspect first 20 lines for a 'p wcnf' or 'p cnf' header
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                s = line.strip().lower()
                if s.startswith('p cnf'):
                    return os.path.abspath(problem_file)
                if s.startswith('p wcnf'):
                    return _convert_wcnf_to_cnf(problem_file, solver_dir)
    except Exception:
        # If file cannot be read for inspection, fall back to extension check
        pass

    if lower.endswith('.cnf'):
        return os.path.abspath(problem_file)
    if lower.endswith('.wcnf'):
        return _convert_wcnf_to_cnf(problem_file, solver_dir)

    # Default: assume CNF
    return os.path.abspath(problem_file)


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
        # Ensure the solver receives a CNF file; convert WCNF to CNF if needed
        problem_path = _ensure_cnf(problem_file, solver_dir_abs)
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
        # Ensure the solver receives a CNF file; convert WCNF to CNF if needed
        problem_path = _ensure_cnf(problem_file, solver_dir_abs)
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
            "FourierSAT_Github_AIJ/FourierSAT.py",
            problem_path,
            "--ismaxsat",
            "1",
            "--timelimit",
            str(timeout),
        ]
        output_lines = _run_solver_command(cmd, timeout, solver_dir_abs)
        print(' '.join(cmd), output_lines)
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



class LMSolver(Solver):
    """Literal-majority baseline solver."""

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        del timeout, init_values, solver_dir
        problem_path = _normalize_problem_path(problem_file)
        cnf = _load_weighted_formula(problem_path)

        has_soft = len(cnf.soft) > 0
        has_hard = len(cnf.hard) > 0

        if has_soft and has_hard:
            raise ValueError("LMSolver supports either SAT or MaxSAT, not both.")

        if has_soft and len(set(cnf.wght)) > 1:
            raise ValueError("LMSolver supports only uniform soft clause weights.")

        clauses = cnf.hard if has_hard else cnf.soft
        n_vars = cnf.nv

        pos_counts = [0] * (n_vars + 1)
        neg_counts = [0] * (n_vars + 1)

        for clause in clauses:
            for literal in clause:
                if literal > 0:
                    pos_counts[literal] += 1
                elif literal < 0:
                    neg_counts[-literal] += 1

        assignment = [0] * n_vars
        for var in range(1, n_vars + 1):
            assignment[var - 1] = 1 if pos_counts[var] >= neg_counts[var] else 0

        cost = 0
        for clause in clauses:
            clause_satisfied = any(
                (lit > 0 and assignment[abs(lit) - 1] == 1)
                or (lit < 0 and assignment[abs(lit) - 1] == 0)
                for lit in clause
            )
            if not clause_satisfied:
                cost += 1
        return cost


class ModelPredictionSolver(Solver):
    """Solver that evaluates raw GNN predictions without refinement."""

    def __init__(
        self,
        model_dir: str,
        model_id: str,
        device=None,
        threshold: float = 0.5,
    ) -> None:
        self.model_dir = model_dir
        self.model_id = model_id
        self.device = device
        self.threshold = threshold
        self._gnn_solver = None
        self.last_accuracy: Optional[float] = None
        self.last_totals: Optional[Tuple[int, int]] = None

    def _get_gnn_solver(self):
        if self._gnn_solver is None:
            from solver.gnn_solver import LSGNNSolver

            self._gnn_solver = LSGNNSolver(
                model_dir=self.model_dir,
                model_id=self.model_id,
                device=self.device,
            )
        return self._gnn_solver

    def solve(
        self,
        problem_file: str,
        timeout: int,
        init_values: Optional[list[float]] = None,
        solver_dir: Optional[str] = None,
    ) -> int:
        del timeout, init_values, solver_dir
        problem_path = _normalize_problem_path(problem_file)
        cnf = _load_weighted_formula(problem_path)

        predictor = self._get_gnn_solver()
        predictions = predictor.predict_initial_values(problem_path)
        if predictions is None:
            print("ModelPredictionSolver: failed to obtain predictions.")
            self.last_accuracy = None
            self.last_totals = None
            return -1

        n_vars = cnf.nv
        assignment = [0] * n_vars
        for idx in range(n_vars):
            value = predictions[idx] if idx < len(predictions) else 0.0
            assignment[idx] = 1 if value >= self.threshold else 0

        clauses = cnf.hard + cnf.soft
        unsatisfied, accuracy = _evaluate_clauses(clauses, assignment)
        total_clauses = len(clauses)
        satisfied = total_clauses - unsatisfied
        self.last_accuracy = accuracy
        self.last_totals = (total_clauses, satisfied)
        if total_clauses > 0:
            print(
                f"ModelPredictionSolver accuracy: {accuracy * 100:.2f}% "
                f"({satisfied}/{total_clauses} clauses satisfied)"
            )
        else:
            print("ModelPredictionSolver accuracy: trivially satisfied (no clauses).")
        return unsatisfied
