import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.nn.conv import MessagePassing

class SAGELA(PyG.SAGEConv):
    def __init__(self, in_channels, out_channels, edge_channels,
                 normalize=False, concat=True, bias=True, **kwargs):
        super(SAGELA, self).__init__(in_channels, out_channels, 
                                         normalize=normalize, concat=concat, bias=bias, **kwargs)
        self.edge_channels = edge_channels
        self.amp_weight = nn.Parameter(torch.Tensor(edge_channels, in_channels))
        self.gate_linear = nn.Linear(2 * in_channels + edge_channels, 1)
        nn.init.xavier_uniform_(self.amp_weight)

    def forward(self, x, edge_index, edge_feature, size=None,
                res_n_id=None):
        if not self.concat and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(self.node_dim))

        return self.propagate(edge_index, size=size, x=x,
                              edge_feature=edge_feature, res_n_id=res_n_id)

    def message(self, x_i, x_j, edge_feature):
        # calculate gate lambda
        lamb_in = torch.cat([x_i, x_j, edge_feature.repeat(x_j.shape[0], 1, 1)], dim=-1)
        lamb = torch.sigmoid(self.gate_linear(lamb_in))

        # amplifier
        amp = torch.matmul(edge_feature, self.amp_weight)
        amp_x_j = amp.view(1, -1, self.in_channels) * x_j

        return amp_x_j * lamb


class SAGELANet(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SAGELANet, self).__init__()
        self.sagela = SAGELA(in_channels, out_channels, edge_channels=1, node_dim=1)
    
    def forward(self, X, edge_index, edge_weight):
        edge_feature = edge_weight.unsqueeze(-1)
        return self.sagela(X, edge_index, edge_feature)
