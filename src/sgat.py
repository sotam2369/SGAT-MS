import typing
from typing import Optional, Tuple, Union

from numpy import positive
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros, ones
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    PairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    scatter
)
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class SGATv2Conv(MessagePassing):
    r"""A GAT-style graph convolutional layer designed for SAT-based graph processing,
    incorporating polarity-aware edge updates to model relationships between variables
    and clauses.

    This layer extends the standard graph attention mechanism by integrating edge polarity
    information, enabling more expressive attention computations in SAT problem graphs.

    Args:
        in_channels (int or Tuple[int, int]): Size of each input sample, or tuple for source and target node features.
        out_channels (int): Size of each output sample per head.
        heads (int, optional): Number of attention heads. (default: 1)
        concat (bool, optional): Whether to concatenate multi-head outputs or average them. (default: True)
        negative_slope (float, optional): LeakyReLU angle of negative slope. (default: 0.2)
        dropout (float, optional): Dropout probability on attention weights. (default: 0.0)
        add_self_loops (bool, optional): Whether to add self-loops to the adjacency matrix. (default: True)
        edge_dim (int, optional): Edge feature dimensionality. (default: None)
        fill_value (float, Tensor, or str, optional): Value to fill self-loop edge features. (default: 'mean')
        bias (bool, optional): If set to False, the layer will not learn an additive bias. (default: True)
        share_weights (bool, optional): Whether to share linear weights for source and target nodes. (default: False)
        residual (bool, optional): Whether to include residual connections. (default: False)
        **kwargs: Additional arguments of MessagePassing.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        r"""Initializes the SGATv2Conv layer.

        Sets up linear transformations, attention parameters, and optional residual
        connections and bias terms.

        Args:
            in_channels (int or Tuple[int, int]): Input feature dimensions.
            out_channels (int): Output feature dimension per head.
            heads (int): Number of attention heads.
            concat (bool): Whether to concatenate or average multi-head outputs.
            negative_slope (float): Negative slope for LeakyReLU activation.
            dropout (float): Dropout rate for attention coefficients.
            add_self_loops (bool): Whether to add self-loops to the graph.
            edge_dim (int, optional): Dimension of edge features.
            fill_value (float, Tensor, or str): Value to fill self-loop edge features.
            bias (bool): Whether to include bias parameters.
            share_weights (bool): Whether to share weights between source and target nodes.
            residual (bool): Whether to add residual connections.
            **kwargs: Additional keyword arguments for MessagePassing.
        """
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=bias,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets learnable parameters of the layer to their initial states."""
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        if hasattr(self, "bias") and self.bias is not None:
            zeros(self.bias)


    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        positive_edges: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the SGATv2Conv layer.

        Args:
            x (Tensor or PairTensor): Input node feature matrix or tuple of source and target features.
            edge_index (Tensor or SparseTensor): Edge indices.
            edge_attr (Tensor, optional): Edge feature matrix. (default: None)
            positive_edges (Tensor, optional): Tensor indicating positive polarity edges. (default: None)
            return_attention_weights (bool, optional): If True, returns attention weights along with output. (default: None)

        Returns:
            Tensor or Tuple: Output node features, optionally with attention weights.
        """
        H, C, C_in = self.heads, self.out_channels, self.in_channels

        assert x.shape[-1] == C_in, f'Feature shape mismatch: {x.shape[-1]} != {C_in}'

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            if C_in != C:
                raise ValueError(
                    f'Input feature size {C_in} does not match output feature size {C}. '
                    'Please set `in_channels` to the correct value.'
                )
            else:
                x_l = x_r = x.unsqueeze(1).repeat(1, H, 1)

        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        alpha = self.edge_updater(edge_index, x=(x, x), edge_attr=edge_attr, positive_edges=positive_edges)

        # propagate_type: (x: PairTensor, alpha: Tensor, positive_edges: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, positive_edges=positive_edges)
        out = torch.clamp(out, min=0, max=1)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        # if self.bias is not None:
        #     out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,#x_aggr_i: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int], positive_edges: OptTensor = None) -> Tensor: # x_i is variable, x_j is clause
        r"""Computes attention scores for edges based on node features and polarity.

        Attention scores are calculated differently depending on the polarity of each edge,
        enabling the model to distinguish between positive and negative clause-variable relations.

        Args:
            x_j (Tensor): Features of target nodes (clauses).
            x_i (Tensor): Features of source nodes (variables).
            edge_attr (Tensor, optional): Edge features.
            index (Tensor): Target node indices for edges.
            ptr (Tensor, optional): Optional pointers for segment-wise operations.
            dim_size (int, optional): Size of the target dimension.
            positive_edges (Tensor, optional): Boolean tensor indicating positive polarity edges.

        Returns:
            Tensor: Normalized attention coefficients for each edge.
        """
        H, C = self.heads, self.out_channels
        
        positive_edges = positive_edges.unsqueeze(-1)
        x = self.lin_l((positive_edges * (1+x_i-x_j) + ~positive_edges * (2-x_i-x_j))).view(-1, H, C)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr_out = self.lin_edge(edge_attr)
            edge_attr_out = edge_attr_out.view(-1, self.heads, self.out_channels)
            x = x + edge_attr_out


        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha_out = softmax(alpha, index, ptr, dim_size)
        return alpha_out


    def message(self, x_j: Tensor, x_i: Tensor, alpha: Tensor, positive_edges: OptTensor = None) -> Tensor: #x_j: Clause, x_i: Variable
        r"""Computes messages passed along edges during propagation, modulated by attention.

        Messages are computed by combining source and target node features with polarity-aware
        adjustments, then weighted by attention coefficients.

        Args:
            x_j (Tensor): Features of neighboring nodes (clauses).
            x_i (Tensor): Features of central nodes (variables).
            alpha (Tensor): Attention coefficients for edges.
            positive_edges (Tensor, optional): Boolean tensor indicating positive polarity edges.

        Returns:
            Tensor: Weighted messages to be aggregated at target nodes.
        """
        positive_edges = positive_edges.unsqueeze(-1).unsqueeze(-1)
        x_j_out = positive_edges * (x_i+1-x_j) + ~positive_edges * (x_i-1+x_j) # Main modifications
        x_j_out = torch.clamp(x_j_out, min=0, max=1)
        x_j_out = alpha.unsqueeze(-1) * x_j_out
        return x_j_out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')







class SGATNorm(torch.nn.Module):
    r"""Sigmoid-based feature normalization layer used in SGAT models.

    This layer normalizes input features to the range [-1, 1] and applies a
    learnable sigmoid transformation centered around a bias term, allowing
    flexible feature scaling and shifting.

    Args:
        in_channels (int): Number of input features.
        eps (float, optional): Small value to avoid numerical instability. (default: 1e-5)
        affine (bool, optional): Whether to learn affine parameters. (default: True)
        mode (str, optional): Normalization mode, e.g., 'graph'. (default: 'graph')
    """
    def __init__(
        self,
        in_channels: int,
    ):
        r"""Initializes the SGATNorm layer.

        Args:
            in_channels (int): Number of input features.
            eps (float): Small epsilon value for numerical stability.
            affine (bool): Whether to learn affine parameters (bias and alpha).
            mode (str): Normalization mode.
        """
        super().__init__()

        self.in_channels = in_channels
        self.bias = Parameter(torch.empty(1, in_channels))
        self.alpha = Parameter(torch.empty(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets learnable parameters to their initial states."""
        zeros(self.bias)
        zeros(self.alpha)

    def forward(self, x: Tensor) -> Tensor:
        r"""Applies sigmoid normalization to input features.

        The input features are first scaled to the range [-1, 1], then transformed
        by a sigmoid function centered around a learnable bias term.

        Args:
            x (Tensor): Input feature tensor.

        Returns:
            Tensor: Normalized output features.
        """

        # Normalize x to [-1, 1] range
        x = x * 2 - 1

        # Apply sigmoid normalization centered around k
        out = torch.sigmoid(self.alpha * (x - self.bias))

        return out
    

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.in_channels})"



class TnormLayer(torch.nn.Module):
    r"""T-norm layer for SGAT models, implementing a t-norm operation on input features.

    This layer applies a t-norm operation to the input features, which is useful for
    combining multiple feature vectors in a way that respects the semantics of the
    underlying problem (e.g., SAT solving).

    Args:
        in_channels (int): Number of input features.
    """
    def __init__(self, t_norm: str = 'product'):
        super().__init__()
        self.t_norm = t_norm

    def forward(self, x: Tensor, edge_index: Tensor, positive_edges: Tensor) -> Tensor:
        r"""Applies the t-norm operation to the input features.

        Args:
            x (Tensor): Input feature tensor.
            edge_index (Tensor): Edge index tensor.
            positive_edges (Tensor): Tensor indicating positive polarity edges.

        Returns:
            Tensor: Output after applying the t-norm operation.
        """
        num_nodes = x.size(0)
        row, col = edge_index
        sgat_pos = positive_edges.bool().unsqueeze(-1)
        x_row = x[row]

        if self.t_norm == 'godel':
            x = torch.where(sgat_pos, x_row, 1 - x_row)
            x_0 = scatter(x, col, dim=0, dim_size=num_nodes, reduce='max')
        elif self.t_norm == 'lukasiewicz':
            x = torch.where(sgat_pos, 1 - x_row, x_row)
            num_clauses = scatter(torch.ones_like(x_row), col, dim=0, dim_size=num_nodes, reduce='sum') - 1
            x_0 = 1 - torch.clamp(scatter(x, col, dim=0, dim_size=num_nodes, reduce='sum') - num_clauses, min=0, max=1)
        elif self.t_norm == 'product':
            x = torch.where(sgat_pos, 1 - x_row, x_row)
            x_0 = 1 - scatter(x, col, dim=0, dim_size=num_nodes, reduce='mul')
        else:
            raise ValueError(f'Invalid t-norm type: {self.t_norm}')

        return x_0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(t_norm='{self.t_norm}')"
