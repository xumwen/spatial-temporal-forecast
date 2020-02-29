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
        
    def forward(self, X, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t1 = X.permute(0, 2, 1, 3).contiguous().view(-1, X.shape[1], X.shape[3])
        t2 = F.relu(self.gcn1(t1, A))
        t3 = torch.sigmoid(self.gcn2(t2, A))
        out = t3.view(X.shape[0], X.shape[2], t3.shape[1], t3.shape[2]).permute(0, 2, 1, 3)

        return out

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, hidden_size)
    
    def forward(self, X):
        """
        :param X: Input data of shape (input_seq_len, batch_size, hidden_size).
        :return output: Output data of shape (input_seq_len, batch_size, hidden_size).
        :return hidden: Last hidden of shape (1, batch_size, hidden_size)
        """
        hidden = torch.zeros(1, X.shape[1], self.hidden_size, device=X.device)
        output, hidden = self.gru(X, hidden)
        return output, hidden

class DecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_seq_len):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        self.gru_cell = torch.nn.GRUCell(input_size, hidden_size)
    
    def forward(self, hidden):
        """
        :param hidden: Encoder's last hidden of shape (batch_size, hidden_size).
        :return: Output data of shape (batch_size, output_seq_len, hidden_size).
        """
        output = torch.zeros(self.output_seq_len, hidden.shape[0], self.hidden_size, device=hidden.device)
        for i in range(self.output_seq_len):
            hidden = self.gru_cell(hidden, hidden)
            output[i] = hidden
        return output.permute(1, 0, 2)
    
class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, input_seq_len, output_seq_len):
        super(GRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        self.encoder_gru = EncoderGRU(input_size, hidden_size)
        self.decoder_gru = DecoderGRU(hidden_size, hidden_size, output_seq_len)
        
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, input_seq_len,
        num_features).
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # batch_size * num_nodes -> batch
        gru_input = X.contiguous().view(-1, X.shape[2], X.shape[3]).permute(1, 0, 2)
        t, hidden = self.encoder_gru(gru_input)
        output = self.decoder_gru(hidden[0])  
        # batch -> batch_size * num_nodes
        output = output.view(X.shape[0], X.shape[1], self.output_seq_len, self.hidden_size)
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
        self.gcn = GCNBlock(in_channels=num_features, spatial_channels=hidden_size,
                                num_nodes=num_nodes, gcn_type=gcn_type)
        self.gru = GRUBlock(input_size=hidden_size, hidden_size=hidden_size,
                                input_seq_len = num_timesteps_input,
                                output_seq_len=num_timesteps_output)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.gcn(X, A)
        out2 = self.gru(out1)
        out3 = self.linear(out2).squeeze(dim = 3)
        return out3
