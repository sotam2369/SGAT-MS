from numpy import positive
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GATConv
import torch

import os
import argparse
import json

from sgat import SGATv2Conv, SGATNorm, TnormLayer


def loadFromID(
    path,
    last=False,
    no_load=False,
    map_location=None,
):
    """
    Load an SGAT model from a specified directory.

    Args:
        path (str): Directory path containing model files and status.json.
        last (bool): If True, load the last saved model state. Otherwise, load the best model.
        no_load (bool): If True, do not load model weights.
        map_location: Device mapping for loading the model.

    Returns:
        SGAT: An instance of the SGAT model with loaded parameters.
    """
    namespace = None
    if os.path.exists(os.path.join(path, "status.json")):
        with open(os.path.join(path, "status.json"), 'r') as f:
            namespace = argparse.Namespace(**json.load(f))
        model = SGAT(
            iterations=int(namespace.layers),
            heads=int(namespace.heads),
            t_norm=namespace.t_norm,
            normalization=namespace.normalization,
            d_edge=int(namespace.edge_dimension),
            hidden=int(namespace.hidden),
            dropout=namespace.dropout,
            use_gat=namespace.use_gat,
        )
    
    if not no_load:
        if last:
            model.load_state_dict(torch.load(os.path.join(path, "last_model.pt"), map_location=map_location))
        else:
            model.load_state_dict(torch.load(os.path.join(path, "best_model.pt"), map_location=map_location))
    return model






class SGAT(nn.Module):
    """
    Self-Gated Attention Transformer (SGAT) model for graph-based learning tasks.

    This model supports both standard GAT and SGATv2 convolutional layers,
    with optional t-norm based pooling and normalization layers. It processes
    graph-structured data with edge attributes and supports iterative message passing.

    Args:
        iterations (int): Number of iterations/layers in the model.
        d_edge (int): Dimension of edge features.
        hidden (int): Hidden feature dimension size.
        heads (int): Number of attention heads.
        bias (bool): Whether to use bias in convolutional layers.
        t_norm (str or None): Type of t-norm for pooling ('product', 'godel', etc.) or None.
        normalization (bool): Whether to apply normalization layers.
        dropout (float): Dropout rate for attention layers.
        use_gat (bool): Whether to use standard GATConv layers instead of SGATv2Conv.
    """
    def __init__(
        self,
        iterations=5,
        d_edge=4,
        hidden=1,
        heads=4,
        bias=True,
        t_norm='product',
        normalization=True,
        dropout=0,
        use_gat=False,
    ):
        super(SGAT, self).__init__()
        self.use_gat = use_gat
        self.iterations = iterations
        self.t_norm = t_norm
        self.d_edge = d_edge
        self.hidden = hidden

        def create_gat_layer(in_channels, out_channels, concat=True):
            class GATWrapper(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gat = GATConv(
                        in_channels,
                        out_channels,
                        heads=heads,
                        concat=concat,
                        bias=bias,
                        dropout=dropout,
                    )
                def forward(self, x, edge_index, edge_attr=None):
                    return self.gat(x, edge_index)
            return GATWrapper()

        def create_sgat_layer(in_channels, out_channels, concat=True):
            return SGATv2Conv(
                in_channels,
                out_channels,
                edge_dim=d_edge,
                heads=heads,
                add_self_loops=False,
                bias=bias,
                concat=concat,
                dropout=dropout,
            )

        gat_hidden_var = []

        self.init_layer = nn.Sequential(nn.Linear(1, self.hidden), nn.Sigmoid())
        if t_norm is not None:
            self.pool_layer = TnormLayer(t_norm)
        
        self.out_pool_layer = TnormLayer('godel')


        # Subsequent layers
        for _ in range(iterations):
            if self.use_gat:
                layer_list = [
                    (create_gat_layer(self.hidden, self.hidden, concat=False), 'x, edge_index, edge_attr -> x'),
                    (nn.Sigmoid(), 'x -> x')
                ]
                if normalization:
                    layer_list.insert(1, (gnn.BatchNorm(self.hidden), 'x -> x'))
                gat_hidden_var.append(gnn.Sequential('x, edge_index, edge_attr', layer_list))
            else:
                layer_list = [
                    (create_sgat_layer(self.hidden, self.hidden, concat=False), 'x, edge_index, edge_attr, positive_edges -> x')
                ]
                if normalization:
                    layer_list.append((SGATNorm(self.hidden), 'x -> x'))
                gat_hidden_var.append(gnn.Sequential('x, edge_index, edge_attr, positive_edges', layer_list))

        self.gat_hidden_var = nn.Sequential(*gat_hidden_var)

        if self.t_norm is None:
            gat_hidden_clause = []
            for _ in range(iterations):
                layers = [
                    (create_gat_layer(self.hidden, self.hidden, concat=False), 'x, edge_index, edge_attr -> x'),
                    (nn.Sigmoid(), 'x -> x')
                ]
                if normalization:
                    layers.insert(1, (gnn.BatchNorm(self.hidden), 'x -> x'))
                gat_hidden_clause.append(gnn.Sequential('x, edge_index, edge_attr', layers))
            self.gat_hidden_clause = nn.Sequential(*gat_hidden_clause)


    def forward(self, data):
        """
        Forward pass of the SGAT model.

        Args:
            data: A data object containing:
                - x (Tensor): Node features tensor.
                - edge_index_clause (Tensor): Edge indices for clauses.
                - edge_attr_clause (Tensor): Edge attributes for clauses.
                - edge_index_var (Tensor): Edge indices for variables.
                - edge_attr_var (Tensor): Edge attributes for variables.
                - mask (Tensor): Boolean mask tensor indicating clause nodes.
                - positive_edges (Tensor): Edge indices for positive edges.
                - batch (Tensor): Batch indices for nodes.

        Returns:
            tuple: (x_out_clauses, x_out_vars, x_out_max)
                - x_out_clauses (Tensor): Output features for clause nodes.
                - x_out_vars (Tensor): Output features for variable nodes.
                - x_out_max (Tensor): Max pooled output features.
        """
        
        x = data.x
        edge_index_clause, edge_attr_clause = data.edge_index_clause, data.edge_attr_clause
        edge_index_var, edge_attr_var = data.edge_index_var, data.edge_attr_var
        mask = data.mask.unsqueeze(-1)
        positive_edges = data.positive_edges
        
        x = self.init_layer(x)

        if self.t_norm is not None:
            x = self.pool_layer(x, edge_index_clause, positive_edges) * mask + x * (1 - mask)

        for i in range(self.iterations):
            if self.use_gat:
                x_var = self.gat_hidden_var[i](x, edge_index_var, edge_attr_var) * (1 - mask)
            else:
                x_var = self.gat_hidden_var[i](x, edge_index_var, edge_attr_var, positive_edges) * (1 - mask)

            if self.t_norm is not None:
                x = self.pool_layer(x_var, edge_index_clause, positive_edges) * mask + x_var
            else:
                x = self.gat_hidden_clause[i](x_var, edge_index_clause, edge_attr_clause) * mask + x_var


        x = torch.mean(x, dim=-1, keepdim=True)
        x = self.out_pool_layer(x, edge_index_clause, positive_edges) * mask + x * (1 - mask)

        x_out_clauses = x[data.mask == 1]
        x_out_vars = x[data.mask == 0]
        x = torch.round(x)
        x_out_max = self.out_pool_layer(x, edge_index_clause, positive_edges) * mask + x * (1 - mask)
        return x_out_clauses, x_out_vars, x_out_max
