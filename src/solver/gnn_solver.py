import os
import time
import math
import logging
from typing import Optional, List, Tuple

import torch
from torch_geometric.loader import DataLoader

from models import loadFromID
from trainer import SGATTrainer
from utils.data import SGATData
from loss import SGATLoss


logger = logging.getLogger(__name__)



class LSGNNSolver:
    """
    Solver using a learned GNN model (SGAT).
    """

    DEFAULT_PATIENCE: int = 10
    RANDOMIZE_PROBS: List[float] = [0.1, 0.2, 0.3, 0.5, 1.0]
    LEARNING_RATE: float = 0.01

    def __init__(self, model_dir: str = "../plots/", model_id: str = "1", device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model_id = model_id
        self.model_path = os.path.join(model_dir, f"train_{model_id}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model directory does not exist: {self.model_path}")
        self.model = loadFromID(self.model_path, no_load=False, last=False, map_location=self.device)

    def predict_initial_values(self, problem_file: str) -> Optional[List[float]]:
        """
        Predict initial values for the given problem file using the SGAT model.

        Args:
            problem_file (str): Path to the problem file.

        Returns:
            Optional[List[float]]: List of initial values or None if prediction fails.
        """
        try:
            data = SGATData(
                file_path=problem_file,
                preload=True,
                device=self.device,
                verbose=False
            )
            data_graph = data.to_data(random_init=True)
            self.model.to(self.device)
            _, assignment, _ = self.model.forward(data_graph)
            return assignment.squeeze().detach().cpu().numpy().tolist()
        except Exception as e:
            logger.info(f"Failed to compute init values from model for {problem_file}: {e}")
            return None

    def estimate_batch_size(self, n_clauses: int) -> int:
        """
        Estimate batch size based on the number of clauses.

        Args:
            n_clauses (int): Number of clauses in the problem.

        Returns:
            int: Estimated batch size.
        """
        if n_clauses < 500:
            return 8
        elif n_clauses < 2000:
            return 4
        elif n_clauses < 10000:
            return 2
        else:
            return 1

    def _update_best(self, current_best: dict, candidate_cost: torch.Tensor, candidate_values: torch.Tensor, epoch: int, n_clauses: int) -> bool:
        """
        Update the current best solution if candidate is better.

        Args:
            current_best (dict): Current best solution dictionary.
            candidate_cost (torch.Tensor): Candidate cost tensor.
            candidate_values (torch.Tensor): Candidate solution values.
            epoch (int): Current epoch number.
            n_clauses (int): Number of clauses.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        def detach_and_cpu(tensor: torch.Tensor) -> torch.Tensor:
            if hasattr(tensor, "detach"):
                return tensor.detach().cpu()
            return tensor

        candidate_cost_float = float(candidate_cost)
        if candidate_cost_float < current_best['cost']:
            current_best.update({
                'epoch': epoch,
                'values': detach_and_cpu(candidate_values),
                'cost': candidate_cost_float
            })
            if candidate_cost_float == 0:
                current_best['early_stop'] = -1
        elif epoch - current_best['epoch'] > current_best.get('early_stop', self.DEFAULT_PATIENCE):
            return True  # Indicates early stopping should occur
        return False

    def _randomize_graph(self, graph: object, randomize_prob: float) -> None:
        """
        Randomize parts of the solution with given probability while keeping fixed variables as final best.

        Args:
            solution (object): Solution object with attributes x and mask.
            randomize_prob (float): Probability of randomizing each variable.
        """
        random_mask = torch.rand_like(graph.x) <= randomize_prob
        graph.x[random_mask] = torch.rand_like(graph.x[random_mask])
        return graph

    def _initialize_batch_state(self, batch_graphs: List[object], n_clauses: int) -> Tuple[List[dict], dict, List[int], List[int]]:
        """
        Initialize the current best solutions and final best solution for the batch.

        Args:
            batch_graphs (List[object]): List of graph data instances.
            n_clauses (int): Number of clauses.

        Returns:
            Tuple containing:
                - current_best (List[dict]): List of best solutions per instance.
                - final_best (dict): The overall best solution.
                - epoch (List[int]): Epoch counters per instance.
                - updated (List[int]): Update counters per instance.
        """
        current_best = []
        for i in range(len(batch_graphs)):
            random_init = torch.rand_like(batch_graphs[i].x)
            batch_graphs[i].x = random_init
            if hasattr(self, 'trainer'):
                inf_out = self.trainer.get_cost(batch_graphs[i], use_weights=True, detach=False)
            else:
                inf_out = SGATTrainer.get_cost(self, batch_graphs[i], use_weights=True, detach=False)  # type: ignore[arg-type]
            current_best.append({
                'epoch': 1,
                'values': inf_out[1][1].detach().cpu() if hasattr(inf_out[1][1], "detach") else inf_out[1][1],
                'cost': float(inf_out[0]),
                'early_stop': self.DEFAULT_PATIENCE
            })
        final_best = min(current_best, key=lambda d: d['cost']).copy()
        epoch = [1] * len(batch_graphs)
        updated = [0] * len(batch_graphs)
        return current_best, final_best, epoch, updated

    def _reset_instance_with_final_best(self, instance_idx: int, elapsed_time: float, current_best: List[dict], final_best: dict, updated: List[int], batch_graphs: List[object], n_clauses: int) -> None:
        """
        Reset the instance's solution using the final best values and update early stopping patience.

        Args:
            instance_idx (int): Index of the instance in the batch.
            elapsed_time (float): Time elapsed since start.
            current_best (List[dict]): List of current best solutions.
            final_best (dict): The overall best solution.
            updated (List[int]): Update counters per instance.
            batch_graphs (List[object]): List of graph data instances.
        """
        if current_best[instance_idx]['cost'] < final_best['cost']:
            logger.info(f"Early stopping (instance {instance_idx})")
            logger.info(f"Time elapsed: {elapsed_time}")
            current_best[instance_idx]['early_stop'] = math.ceil(
                current_best[instance_idx]['early_stop'] * (final_best['cost'] + 2) / (current_best[instance_idx]['cost'] + 1)
            )
            final_best.update(current_best[instance_idx])
            logger.info(f"New best found: {final_best['cost']}")
            logger.info(f"Early stopping patience: {current_best[instance_idx]['early_stop']}\n")
            updated[instance_idx] = 0
        if updated[instance_idx] == len(self.RANDOMIZE_PROBS):
            updated[instance_idx] = 0
            current_best[instance_idx]['early_stop'] = self.DEFAULT_PATIENCE

        batch_graphs[instance_idx] = self._randomize_graph(batch_graphs[instance_idx], self.RANDOMIZE_PROBS[updated[instance_idx]])
        self.trainer.train_loader = DataLoader(batch_graphs, batch_size=len(batch_graphs), shuffle=False)
        current_best[instance_idx]['cost'] = self.sum_weights + 1
        updated[instance_idx] += 1
        # Reset epoch count for this instance
        # Note: epoch is incremented in main loop after this call

    def solve(self, problem_file: str, timeout: float, init_values: Optional[List[float]] = None) -> int:
        """
        Solve the problem using the learned GNN model.

        Args:
            problem_file (str): Path to the problem file.
            timeout (float): Time limit in seconds.
            init_values (Optional[List[float]]): Initial values for the solver.

        Returns:
            int: The cost of the best solution found.
        """
        time_start = time.perf_counter()
        
        # Prepare data using SGATData
        cnf_data = SGATData(
            file_path=problem_file,
            preload=True,
            device=self.device,
            verbose=False
        )

        n_clauses = getattr(cnf_data, "n_clauses", 1000)
        batch_size = self.estimate_batch_size(n_clauses)
        batch_graphs = [cnf_data.to_data(random_init=True) for _ in range(batch_size)]
        optimizer = torch.optim.Adam
        loss = SGATLoss(device=self.device)
        trainer = SGATTrainer(
            model=self.model,
            optimizer_cls=optimizer,
            device=self.device,
            train_loader=DataLoader(batch_graphs, batch_size=batch_size, shuffle=False),
            val_loader=None,
            loss_fn=loss,
            learning_rate=self.LEARNING_RATE,
            randomize_input_prob=0,
            cut_incomplete_batch=False
        )
        self.trainer = trainer
        self.sum_weights = cnf_data.weights.sum().item()

        # Pre-training step
        trainer.pre_training()


        current_best, final_best, epoch, updated = self._initialize_batch_state(batch_graphs, n_clauses)

        while True:
            elapsed_time = time.perf_counter() - time_start
            if elapsed_time > timeout or final_best['cost'] == 0:
                break

            _, _, outputs, costs = trainer.train_epoch(compute_cost=True)

            early_stop_triggered_list = []
            for instance_idx, (cost, candidate_values) in enumerate(zip(costs, outputs)):
                if self._update_best(current_best[instance_idx], cost, candidate_values, epoch[instance_idx], n_clauses):
                    early_stop_triggered_list.append(True)
                else:
                    early_stop_triggered_list.append(False)

            for instance_idx, triggered in enumerate(early_stop_triggered_list):
                if not triggered:
                    epoch[instance_idx] += 1
                    continue
                self._reset_instance_with_final_best(instance_idx, elapsed_time, current_best, final_best, updated, batch_graphs, n_clauses)
                epoch[instance_idx] = 1

        for instance_idx in range(batch_size):
            if current_best[instance_idx]['epoch'] is None or current_best[instance_idx]['cost'] < final_best['cost']:
                final_best = current_best[instance_idx]
        final_best['cost'] = int(final_best['cost'])
        min_cost = final_best['cost']
        logger.info(f"Final best found: {min_cost}")
        logger.info(f"Time elapsed: {time.perf_counter() - time_start}")
        return min_cost
