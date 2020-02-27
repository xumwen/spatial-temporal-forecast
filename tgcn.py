import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GCNConv, ChebConv

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
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        
    def forward(self, X, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        bn = self.batch_norm(X)
        t1 = bn.permute(0, 2, 1, 3).contiguous().view(-1, bn.shape[1], bn.shape[3])
        t2 = self.gcn1(t1, A)
        gcn1 = t2.view(X.shape[0], X.shape[2], t2.shape[1], t2.shape[2]).permute(0, 2, 1, 3)
        relu1 = F.relu(gcn1)
        
        t3 = relu1.permute(0, 2, 1, 3).contiguous().view(-1, relu1.shape[1], relu1.shape[3])
        t4 = self.gcn2(t3, A)
        gcn2 = t4.view(X.shape[0], X.shape[2], t4.shape[1], t4.shape[2]).permute(0, 2, 1, 3)
 
        output = torch.sigmoid(gcn2)
        return output

class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_seq_len):
        super(GRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        self.gru = torch.nn.GRU(input_size, hidden_size)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, input_seq_len,
        num_features).
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # batch_size * num_nodes -> batch
        gru_input = X.view(-1, X.shape[2], X.shape[3]).permute(1, 0, 2)
        hidden = torch.zeros(1, X.shape[0] * X.shape[1], self.hidden_size, device=X.device)
        t, hidden = self.gru(gru_input, hidden)
        output = torch.zeros(self.output_seq_len, gru_input.shape[1], self.hidden_size, device=X.device)
        for i in range(self.output_seq_len):
            t, hidden = self.gru(t, hidden)
            output[i] = t[0]
        # batch_size -> (batch_size, num_nodes)
        output = output.permute(1, 0, 2).view(X.shape[0], X.shape[1], self.output_seq_len, self.hidden_size)
        return output

class TGCN(nn.Module):
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
        super(TGCN, self).__init__()
        # self.input_linear = nn.Linear(num_features, hidden_size)
        self.gcn = GCNBlock(in_channels=num_features, spatial_channels=hidden_size,
                                num_nodes=num_nodes, gcn_type=gcn_type)
        self.gru = GRUBlock(input_size=hidden_size, hidden_size=hidden_size,
                                output_seq_len=num_timesteps_output)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.gcn(X, A)
        # out1 = self.input_linear(X)
        out2 = self.gru(out1)
        out3 = self.linear(out2).squeeze(dim = 3)
        return out3
