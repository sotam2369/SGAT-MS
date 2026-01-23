import os
from pysat.formula import CNFPlus, WCNF
from torch_geometric.data import Data
import torch
import signal
from tqdm import tqdm


class SGATData:
    """
    Data structure for handling CNF/WCNF formulas and their graph representations,
    with support for weighted formulas, feature normalization, and various configuration flags.
    """
    def __init__(
        self,
        file_path: str,
        *,
        preload: bool = False,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ):
        # --- Compute device ---
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        # --- File paths ---
        self.file_path: str = file_path
        self.file_name: str = os.path.basename(file_path)

        # --- Configuration flags ---
        self.verbose: bool = verbose

        # --- Runtime state ---
        self.formula = None
        # The following attributes will be set after loading:
        # self.weights, self.ori_weights, self.n_vars, self.n_clauses, etc.

        if preload:
            self._preload_formula()

    def _preload_formula(self) -> None:
        """Helper to preload the formula and print file status."""
        self.quickLoad()
        self.sum_weights = self.weights.sum()
        print("\n=== Formula Summary ===")
        print(f"Variables    : {self.n_vars}")
        print(f"Sum Weights  : {self.sum_weights:.2f}")
        print("========================\n")


    def quickLoad(self, custom_soft: tuple = None) -> None:
        """
        Quickly loads WCNF or CNF file and computes edge features for the current simplified structure.
        - Only uses self.file_path, and self.edge_features.
        - No normalization, no clause statistics, no legacy flags.
        """
        if self.file_path.endswith(".cnf"):
            weights, clause_list_flat, clause_list_flat_num, pos_clause_count, neg_clause_count, n_vars, n_clauses = self.parse_cnf_file()
        else:
            with open(self.file_path, "r") as f:
                lines = f.readlines()
            if self.verbose:
                iter_lines = tqdm(lines, desc="Parsing WCNF", disable=not self.verbose)
            else:
                iter_lines = lines
            weights, clause_list_flat, clause_list_flat_num, pos_clause_count, neg_clause_count, n_vars, n_clauses = self._parse_wcnf_lines(iter_lines)
        self.pos_clause_count = torch.tensor(pos_clause_count, device=self.device)
        self.neg_clause_count = torch.tensor(neg_clause_count, device=self.device)

        # 1. Use torch for all arrays/tensors
        if custom_soft is not None:
            self.weights = custom_soft[0]
        else:
            self.weights = torch.tensor(weights, dtype=torch.float, device=self.device)

        self.n_vars = n_vars
        self.n_clauses = n_clauses

        # 2. Use torch for clause_list_flat and clause_list_flat_num
        clause_list_flat = torch.tensor(clause_list_flat, dtype=torch.long, device=self.device)
        clause_list_flat_num = torch.tensor(clause_list_flat_num, dtype=torch.long, device=self.device)
        self.clause_list_flat_num = clause_list_flat_num

        # 4. Compute edge attributes (inputs are now already torch tensors)
        edge_attr_var, edge_attr_clause, polarity = self._compute_edge_attributes(
            clause_list_flat, clause_list_flat_num
        )
        self.edge_attr_var = edge_attr_var
        self.edge_attr_clause = edge_attr_clause
        self.positive_edges = (polarity == 0)

        # 5. Use torch.cat for edge indices
        self.edge_index_clause = torch.cat([
            (clause_list_flat.abs() - 1).unsqueeze(0),
            (self.n_vars + clause_list_flat_num).unsqueeze(0)
        ], dim=0)

        self.edge_index_var = torch.cat([
            (self.n_vars + clause_list_flat_num).unsqueeze(0),
            (clause_list_flat.abs() - 1).unsqueeze(0)
        ], dim=0)


    def parse_cnf_file(self) -> tuple[list[float], list[int], list[int], list[int], list[int], int, int]:
        """
        Parses an unweighted CNF file and returns the same tuple as _parse_wcnf_lines:
        weights, lits_flat, idxs, pos_cnt, neg_cnt, n_vars, n_clauses
        """
        from pysat.formula import CNFPlus
        formula = CNFPlus(from_file=self.file_path)
        clauses = formula.clauses
        weights = [1.0] * len(clauses)
        lits_flat, idxs, pos_cnt, neg_cnt = [], [], [], []
        n_vars = formula.nv
        for ci, clause in enumerate(clauses):
            lits_flat.extend(clause)
            idxs.extend([ci] * len(clause))
            pos_cnt.append(sum(l > 0 for l in clause))
            neg_cnt.append(sum(l < 0 for l in clause))
        n_clauses = len(clauses)
        return weights, lits_flat, idxs, pos_cnt, neg_cnt, n_vars, n_clauses


    def _compute_edge_attributes(
        self,
        lits,
        clause_idx
    ):
        """
        Compute edge_attr_var and edge_attr_clause for the graph using PyTorch on the correct device.
        Args:
            clause_list_flat: torch.Tensor of flattened literals from all clauses
            clause_list_flat_num: torch.Tensor of clause indices corresponding to each literal
            var_count: torch.Tensor of variable counts (for denominator)
            clause_count: torch.Tensor of clause counts (for denominator)
        Returns:
            edge_attr_var, edge_attr_clause: torch.Tensors for edge attributes
        """

        # Load edge features into a tensor
        edge_features = torch.tensor(
            [[[1,0,0,0], [0,1,0,0]], [[0,0,1,0], [0,0,0,1]]],
            device=self.device,
            dtype=torch.float
        )

        # Prepare clause count inverses
        pos_counts = self.pos_clause_count.clone().detach().float()
        neg_counts = self.neg_clause_count.clone().detach().float()
        pos_inv = torch.where(pos_counts > 0, 1.0 / pos_counts, torch.zeros_like(pos_counts))
        neg_inv = torch.where(neg_counts > 0, 1.0 / neg_counts, torch.zeros_like(neg_counts))

        # Gather per-edge weights
        pw = pos_inv[clause_idx].unsqueeze(1)
        nw = neg_inv[clause_idx].unsqueeze(1)

        # Determine literal polarity (0=pos, 1=neg)
        polarity = (lits < 0).long()

        # Select base features for var-to-clause edges
        feat_pos = edge_features[polarity, 0]
        feat_neg = edge_features[polarity, 1]

        # Combine features according to counts
        base_attr = feat_pos * pw + feat_neg * nw

        # Normalize by clause weight
        wmax = self.weights.max().clamp_min(1e-8)
        weight_factor = self.weights[clause_idx].unsqueeze(1) / wmax
        edge_attr_var = base_attr * weight_factor

        # TODO: Explore additional clause-specific edge features here

        edge_attr_clause = edge_attr_var.clone()

        return edge_attr_var, edge_attr_clause, polarity


    def _parse_wcnf_lines(self, lines) -> tuple[list[float], list[int], list[int], list[int], list[int], int, int]:
        weights = []
        lits_flat, idxs = [], []
        pos_cnt, neg_cnt = [], []
        n_vars = 0
        n_clauses = 0  # will be set to clause_counter at end
        clause_counter = 0

        for line in lines:
            s = line.strip()
            if not s or s.startswith('c'):
                continue
            parts = s.split()
            if parts[0] == 'p':
                n_vars = int(parts[-3])
                # ignore header n_clauses; will count real clauses
                continue
            lits = list(map(int, parts[1:-1]))
            token = parts[0]
            try:
                weight = float(token)
            except ValueError:
                weight = 1.0
            weights.append(weight)
            count = len(lits)
            lits_flat.extend(lits)
            idxs.extend([clause_counter] * count)
            pos_cnt.append(sum(l > 0 for l in lits))
            neg_cnt.append(sum(l < 0 for l in lits))
            if lits:
                m = max(abs(l) for l in lits)
                if m > n_vars:
                    n_vars = m
            clause_counter += 1

        n_clauses = clause_counter
        return weights, lits_flat, idxs, pos_cnt, neg_cnt, n_vars, n_clauses



    def to_data(self, random_init: bool = False) -> Data:
        """
        Converts the loaded WCNF/CNF instance into a PyTorch Geometric Data object.

        Args:
            random_init (bool): If True, initialize variable features randomly. Otherwise, set to 0.5.

        Returns:
            Data: Torch Geometric Data object containing graph structure and features.
        """
        mask = torch.cat([
            torch.zeros(self.n_vars, dtype=torch.bool, device=self.device),
            torch.ones(self.n_clauses, dtype=torch.bool, device=self.device)
        ])

        lit_features = torch.rand((self.n_vars, 1), device=self.device) if random_init \
            else torch.full((self.n_vars, 1), 0.5, device=self.device)
        clause_features = torch.full((self.n_clauses, 1), 0.5, device=self.device)
        features = torch.cat([lit_features, clause_features], dim=0)

        data_kwargs = {
            "x": features,
            "edge_index_var": self.edge_index_var,
            "edge_index_clause": self.edge_index_clause,
            "mask": mask.float(),
            "edge_attr_var": self.edge_attr_var,
            "edge_attr_clause": self.edge_attr_clause,
            "weights": self.weights,
            "positive_edges": self.positive_edges,
            "name": self.file_name,
        }

        return Data(**data_kwargs)
    


def main():
    import argparse
    import time
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the .wcnf file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu or cuda")
    parser.add_argument("--verbose", action="store_true", help="Show progress with tqdm")

    args = parser.parse_args()

    data = SGATData(
        file_path=args.file_path,
        preload=True,
        device=torch.device(args.device),
        verbose=args.verbose
    )

    print("Edge index var shape:", data.edge_index_var.shape)
    print("Edge attr var shape:", data.edge_attr_var.shape)
    print("Edge index clause shape:", data.edge_index_clause.shape)
    print("Edge attr clause shape:", data.edge_attr_clause.shape)
    print("Top 10 edge attr var:", data.edge_attr_var[:10])
    print("Bottom 10 edge attr var:", data.edge_attr_var[-10:])
    print("Loaded CNFData successfully.")
    print(f"Execution Time: {time.time() - start:.4f} seconds")

if __name__ == "__main__":
    main()
