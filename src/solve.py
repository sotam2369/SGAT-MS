import argparse
import os
import sys
import random
import numpy as np
import atexit
import torch
import threading

# Import solvers
from solver.solver_wrapper import (
    NuWLSSolver,
    MixingSolver,
    MixSATSolver,
    BandHSSolver,
    SATLikeSolver,
    FourierSATSolver,
    SPBSolver,
    LMSolver,
    ModelPredictionSolver,
)
from solver.gnn_solver import LSGNNSolver

def parse_args():
    parser = argparse.ArgumentParser(description="General Solver Launcher")
    parser.add_argument("--solver", type=str, required=True,
                        choices=["sgat", "nuwls", "mixing", "mixsat", "bandhs", "satlike3.0", "fouriersat", "spb", "lm", "model-predict"],
                        help="Solver to use")
    parser.add_argument("--problem", type=str, required=True, help="Problem file path")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--init", action="store_true", help="Use SGATData to extract init values")
    parser.add_argument("--init-source", type=str, default=None, help="Path to file to load init values from (SGATData), defaults to --problem")
    parser.add_argument("--print-init", action="store_true", help="Print initial values and exit")
    # New flags for SGAT model pretraining
    parser.add_argument("--train", action="store_true", help="Run SGAT pretraining before solving")
    parser.add_argument("--model-dir", type=str, default="../plots/", help="Directory containing SGAT models")
    parser.add_argument("--model-id", type=str, default="1", help="SGAT model ID")
    parser.add_argument("--save-cost-path", type=str, default=None, help="Path to file where cost will be saved")
    parser.add_argument("--solver-dir", type=str, default=".", help="Directory containing the solver executables")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA device")
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def main():
    args = parse_args()
    args.save_cost_path = os.path.abspath(args.save_cost_path) if args.save_cost_path else None
    problem_path = os.path.abspath(args.problem)
    solver_dir = os.path.abspath(args.solver_dir or ".")
    set_seeds(args.seed)

    if args.cuda is not None:
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_entry = {"problem": problem_path, "cost": -1.0}
    placeholder_written = False
    # If this is set to True we will NOT remove the placeholder in the atexit cleanup.
    # This is used when the process exits due to a CUDA OOM so the placeholder remains
    # for later inspection / rerun.
    keep_placeholder_on_oom = False

    # Register an exception hook so that if an uncaught exception (including
    # CUDA OOM) terminates the process, we can mark the placeholder to be
    # kept instead of being removed by atexit cleanup. We also wrap
    # threading.excepthook when available to catch uncaught exceptions in
    # threads.
    original_excepthook = sys.excepthook

    def _mark_oom_from_exception(exc_type, exc_value):
        # Helper that inspects an exception and sets the keep flag when it
        # appears to be a CUDA out-of-memory error.
        nonlocal keep_placeholder_on_oom
        try:
            import torch as _torch
            oom_type = getattr(_torch.cuda, 'OutOfMemoryError', None)
            if oom_type is not None and issubclass(exc_type, oom_type):
                keep_placeholder_on_oom = True
                return True
        except Exception:
            pass
        try:
            if exc_value is not None and 'out of memory' in str(exc_value).lower():
                keep_placeholder_on_oom = True
                return True
        except Exception:
            pass
        return False

    def exception_hook(exc_type, exc_value, exc_tb):
        # Called for uncaught exceptions in the main thread.
        try:
            if _mark_oom_from_exception(exc_type, exc_value):
                print("Uncaught exception detected as CUDA OOM; keeping placeholder.")
        except Exception:
            pass
        try:
            original_excepthook(exc_type, exc_value, exc_tb)
        except Exception:
            # Be defensive: don't let the hook raise.
            pass

    sys.excepthook = exception_hook

    # If available, wrap threading.excepthook (Python 3.8+) to catch OOMs in
    # worker threads as well.
    if hasattr(threading, 'excepthook'):
        original_thread_excepthook = threading.excepthook

        def thread_excepthook(args):
            try:
                _mark_oom_from_exception(args.exc_type, args.exc_value)
            except Exception:
                pass
            try:
                original_thread_excepthook(args)
            except Exception:
                pass

        threading.excepthook = thread_excepthook

    def cleanup_placeholder():
        # Only remove the placeholder on premature exit if we are NOT explicitly
        # keeping it (e.g. when a CUDA OOM occurred).
        if placeholder_written and not keep_placeholder_on_oom and args.save_cost_path and os.path.exists(args.save_cost_path):
            import pandas as pd
            df = pd.read_csv(args.save_cost_path)
            df = df[df["problem"] != result_entry["problem"]]
            df.to_csv(args.save_cost_path, index=False)
            print("Removed placeholder due to premature exit.")

    atexit.register(cleanup_placeholder)

    # Early exit if problem already exists in save_cost_path
    if args.save_cost_path and os.path.exists(args.save_cost_path):
        import pandas as pd
        existing_df = pd.read_csv(args.save_cost_path)
        # Skip if the exact problem path is present OR if any existing entry has
        # the same filename (basename). This handles cases where the same file
        # may be referenced via different paths but should be considered the
        # same instance.
        try:
            existing_problems = existing_df["problem"].dropna().astype(str).tolist()
        except Exception:
            existing_problems = []
        existing_basenames = {os.path.basename(p) for p in existing_problems}
        if problem_path in existing_problems or os.path.basename(problem_path) in existing_basenames:
            print(f"Problem {problem_path} already exists in {args.save_cost_path} (matched by path or filename). Skipping.")
            return

    # Write placeholder result if save_cost_path is provided
    if args.save_cost_path:
        import pandas as pd
        os.makedirs(os.path.dirname(args.save_cost_path), exist_ok=True)
        if os.path.exists(args.save_cost_path):
            df = pd.read_csv(args.save_cost_path)
        else:
            df = pd.DataFrame(columns=["problem", "cost"])
        for key in result_entry.keys():
            if key not in df.columns:
                df[key] = np.nan
        df.loc[len(df)] = {col: result_entry.get(col, np.nan) for col in df.columns}
        df.to_csv(args.save_cost_path, index=False)
        placeholder_written = True

    # Error checking
    assert os.path.exists(problem_path), f"Problem file does not exist: {problem_path}"

    # If training is enabled, use LSGNNSolver to solve the problem
    cost = None
    if args.train:
        solver = LSGNNSolver(model_dir=args.model_dir, model_id=args.model_id, device=device)
        cost = solver.solve(problem_path, args.timeout)
    else:
        solver_factories = {
            "nuwls": NuWLSSolver,
            "mixing": MixingSolver,
            "mixsat": MixSATSolver,
            "bandhs": BandHSSolver,
            "satlike3.0": SATLikeSolver,
            "fouriersat": FourierSATSolver,
            "spb": SPBSolver,
            "lm": LMSolver,
            "model-predict": lambda: ModelPredictionSolver(
                model_dir=args.model_dir,
                model_id=args.model_id,
                device=device,
            ),
        }

        if args.solver == "sgat":
            solver_instance = LSGNNSolver(model_dir=args.model_dir, model_id=args.model_id, device=device)
        else:
            if args.solver not in solver_factories:
                raise ValueError(f"Unknown solver: {args.solver}")
            solver_instance = solver_factories[args.solver]()

        supported_init = {"nuwls", "bandhs", "satlike3.0", "spb"}
        use_init = args.init and args.solver in supported_init
        if args.init and not use_init:
            print(f"--init is not supported for solver {args.solver}. Ignoring.")

        should_prepare_init = use_init or args.print_init
        init_values = None
        if should_prepare_init:
            problem_for_init = os.path.abspath(args.init_source) if args.init_source else problem_path
            init_solver = (
                solver_instance if isinstance(solver_instance, LSGNNSolver)
                else LSGNNSolver(model_dir=args.model_dir, model_id=args.model_id, device=device)
            )
            init_values = init_solver.predict_initial_values(problem_for_init)
            if args.print_init:
                print(f"Initial values: {init_values}")
                sys.exit(0)

        # Print configuration
        print("\n=== Solver Configuration ===")
        print(f"Solver       : {args.solver}")
        print(f"Problem File : {problem_path}")
        print(f"Solver Dir   : {solver_dir}")
        print(f"Timeout (s)  : {args.timeout}")
        print(f"Seed         : {args.seed}")
        print(f"Model Dir    : {args.model_dir}")
        print(f"Model ID     : {args.model_id}")
        print(f"Train SGAT   : {args.train}")
        print(f"Use SGAT Init: {use_init}")
        print(f"Save Path    : {args.save_cost_path}")
        if init_values is not None:
            detail = "Loaded" if use_init else "Computed"
            print(f"{detail} {len(init_values)} initial values")
        print("=============================\n")

        try:
            if isinstance(solver_instance, LSGNNSolver):
                cost = solver_instance.solve(
                    problem_path,
                    args.timeout,
                    init_values=init_values if use_init else None,
                )
            else:
                cost = solver_instance.solve(
                    problem_path,
                    args.timeout,
                    init_values=init_values if use_init else None,
                    solver_dir=solver_dir,
                )
        except Exception as e:
            # Detect CUDA out-of-memory errors. There are a few different ways
            # these can present (specific exception types or generic RuntimeError
            # messages containing 'out of memory'). If we hit an OOM we want to
            # exit early but leave the placeholder row in the save CSV for
            # later inspection or rerun.
            is_oom = False
            try:
                import torch as _torch
                # torch.cuda.OutOfMemoryError may be available on some PyTorch versions
                oom_type = getattr(_torch.cuda, 'OutOfMemoryError', None)
                if oom_type is not None and isinstance(e, oom_type):
                    is_oom = True
            except Exception:
                pass

            if not is_oom:
                # Fallback: look for 'out of memory' in the message text
                try:
                    if 'out of memory' in str(e).lower():
                        is_oom = True
                except Exception:
                    pass

            if is_oom:
                print("CUDA out of memory detected while running the solver. Leaving placeholder intact and exiting.")
                # Prevent the atexit cleanup from removing the placeholder.
                keep_placeholder_on_oom = True
                # Exit; atexit will run but cleanup_placeholder will skip removal.
                sys.exit(1)
            else:
                print(f"Error running solver: {e}")
                cost = -1
    
    print(f"Result: {cost}")
    if args.save_cost_path:
        import pandas as pd
        df = pd.read_csv(args.save_cost_path)
        df.loc[df["problem"] == result_entry["problem"], "cost"] = cost
        df.to_csv(args.save_cost_path, index=False)
        print(f"Updated result in {args.save_cost_path}")
        placeholder_written = False  # Prevent atexit from removing it


if __name__ == "__main__":
    main()
