import argparse
import os
import sys
import random
import numpy as np
import atexit
import torch

# Import solvers
from solver.solver_wrapper import (
    NuWLSSolver,
    MixingSolver,
    MixSATSolver,
    BandHSSolver,
    SATLikeSolver,
    FourierSATSolver,
    SPBSolver,
)
from solver.gnn_solver import LSGNNSolver

def parse_args():
    parser = argparse.ArgumentParser(description="General Solver Launcher")
    parser.add_argument("--solver", type=str, required=True,
                        choices=["sgat", "nuwls", "mixing", "mixsat", "bandhs", "satlike3.0", "fouriersat", "spb"],
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

    def cleanup_placeholder():
        if placeholder_written and args.save_cost_path and os.path.exists(args.save_cost_path):
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
        if problem_path in existing_df["problem"].values:
            print(f"Problem {problem_path} already exists in {args.save_cost_path}. Skipping.")
            return

    # Write placeholder result if save_cost_path is provided
    if args.save_cost_path:
        import pandas as pd
        os.makedirs(os.path.dirname(args.save_cost_path), exist_ok=True)
        if os.path.exists(args.save_cost_path):
            df = pd.read_csv(args.save_cost_path)
        else:
            df = pd.DataFrame(columns=["problem", "cost"])
        df = pd.concat([df, pd.DataFrame([result_entry])], ignore_index=True)
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
