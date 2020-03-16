import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GCNConv, ChebConv
from kencoder import KRNN


class GCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, num_nodes, gcn_type):
        super(GCNBlock, self).__init__()
        if gcn_type == 'cheb':
            GCNCell = ChebConv
        else:
            GCNCell = GCNConv
        self.gcn1 = GCNCell(in_channels=in_channels,
                            out_channels=spatial_channels)
        self.gcn2 = GCNCell(in_channels=spatial_channels,
                            out_channels=spatial_channels)

    def forward(self, X, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t1 = X.permute(0, 2, 1, 3).contiguous(
        ).view(-1, X.shape[1], X.shape[3])
        t2 = F.leaky_relu(self.gcn1(t1, A))
        t3 = F.leaky_relu(self.gcn2(t2, A))
        out = t3.view(X.shape[0], X.shape[2], t3.shape[1],
                      t3.shape[2]).permute(0, 2, 1, 3)

        return out


class FTGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(FTGCN, self).__init__()
        self.gru = KRNN(num_nodes, num_features, num_timesteps_input, hidden_size)
        self.gcn = GCNBlock(in_channels=hidden_size, spatial_channels=hidden_size,
                            num_nodes=num_nodes, gcn_type=gcn_type)
        self.linear = nn.Linear(hidden_size, num_timesteps_output)

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.gru(X)
        out1 = out1.unsqueeze(dim=2)
        out2 = self.gcn(out1, A)

        out2 = out2.squeeze(dim=2)
        out2 = self.linear(out2)

        return out2
