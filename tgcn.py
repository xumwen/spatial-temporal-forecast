import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, num_nodes):
        super(GCNBlock, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels,
                                                     spatial_channels))
        self.Theta2 = nn.Parameter(torch.FloatTensor(spatial_channels,
                                                     spatial_channels))
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()
        
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        bn1 = self.batch_norm(X)
        gcn1 = torch.einsum("ij,jklm->kilm", [A_hat, bn1.permute(1, 0, 2, 3)])
        relu1 = F.relu(torch.matmul(gcn1, self.Theta1))
        bn2 = self.batch_norm(relu1)
        gcn2 = torch.einsum("ij,jklm->kilm", [A_hat, bn2.permute(1, 0, 2, 3)])
        output = torch.sigmoid(torch.matmul(gcn2, self.Theta2))
        
        return output

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.Theta2.shape[1])
        self.Theta2.data.uniform_(-stdv2, stdv2)

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
                 num_timesteps_output, hidden_size = 64):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(TGCN, self).__init__()
        #self.input_linear = nn.Linear(num_features, hidden_size)
        self.gcn = GCNBlock(in_channels = num_features, spatial_channels = hidden_size,
                                num_nodes = num_nodes)
        self.gru = GRUBlock(input_size = hidden_size, hidden_size = hidden_size,
                                output_seq_len = num_timesteps_output)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.gcn(X, A_hat)
        #out1 = self.input_linear(X)
        out2 = self.gru(out1)
        out3 = self.linear(out2).squeeze(dim = 3)
        return out3
