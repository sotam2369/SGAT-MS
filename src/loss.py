import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class SGATLoss(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        loss_type: nn.Module = nn.MSELoss,
        **_,
    ):
        super().__init__()
        self.device = device
        self.loss_func = loss_type(reduction="none")

    def forward(self, output: list[torch.Tensor], data) -> torch.Tensor:
        """
        Computes the clause reconstruction loss for SGAT.

        Args:
            output (list[Tensor]): GNN outputs.
            data: Batch graph data containing clause and variable masks.

        Returns:
            Tensor: Loss tensor of shape (num_graphs,) containing weighted clause losses per graph.
        """
        clause_mask = data.mask == 1
        batch_clause = data.batch[clause_mask]

        out_clause = output[0][:, 0]
        target = torch.ones_like(out_clause)
        loss_elem = self.loss_func(out_clause, target)

        weights = data.weights.to(loss_elem.device)
        weighted_sum = scatter(loss_elem * weights, batch_clause, dim=0, dim_size=data.num_graphs, reduce="sum")
        weight_total = scatter(weights, batch_clause, dim=0, dim_size=data.num_graphs, reduce="sum")
        return weighted_sum / weight_total

    def evaluation(self, output: list[torch.Tensor], data) -> torch.Tensor:
        """
        Evaluates model output against clause satisfaction metrics.

        Args:
            output (list[Tensor]): GNN outputs.
            data: Batch graph data.

        Returns:
            Tensor: Scalar tensor containing the weighted clause satisfaction averaged across graphs.
        """
        clause_mask = data.mask == 1
        out_clause = torch.round(output[2][clause_mask, 0])
        batch = data.batch[clause_mask]
        weights = data.weights.to(out_clause.device)

        weighted_sat = scatter(out_clause * weights, batch, dim=0, dim_size=data.num_graphs, reduce="sum")
        weight_total = scatter(weights, batch, dim=0, dim_size=data.num_graphs, reduce="sum")

        graph_scores = weighted_sat / weight_total
        return graph_scores.mean()
