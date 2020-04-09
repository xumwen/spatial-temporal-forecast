import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """
    The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    """
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: Number of input features at each node.
        :param out_channels: Desired number of output channels at each node.
        """
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels,
                                                     out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)

    def norm(self, A):
        """
        Returns the degree normalized adjacency matrix.
        Formula: A_wave = D_hat^(-1/2) * A_hat * D_hat^(-1/2), and A_hat = A + I
        """
        A_hat = A + torch.diag(torch.ones(A.shape[0])).to(A.device)
        D = A_hat.sum(1).view((-1,))
        D[D <= 10e-5] = 10e-5
        diag = torch.reciprocal(torch.sqrt(D))
        A_wave = diag.view((-1, 1)) * A_hat * diag.view((1, -1))
        
        return A_wave
    
    def forward(self, X, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        A_wave = self.norm(A)
        t = torch.einsum("ij,jkl->kil", [A_wave, X.permute(1, 0, 2)])
        out = torch.matmul(t, self.weight)
        
        return out


class ChebConv(nn.Module):
    """
    The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper
    """
    def __init__(self, in_channels, out_channels, K=3):
        """
        :param in_channels: Number of input features at each node.
        :param out_channels: Desired number of output channels at each node.
        :param K: Chebyshev filter size, i.e. number of hops 𝐾.
        """
        super(ChebConv, self).__init__()
        assert K > 0
        
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[2])
        self.weight.data.uniform_(-stdv, stdv)
       
    def norm(self, A, lambda_max):
        """
        Returns the normalized and scaled adjacency matrix.
        Formula: L = I - D^(-1/2) * A * D^(-1/2), L_hat = (2L / lambda_max) - I
        """
        I = torch.diag(torch.ones(A.shape[0])).to(A.device)
        # Remove self-loops
        A = A - A[0][0] * I
        
        D = A.sum(1).view((-1,))
        D[D <= 10e-5] = 10e-5
        diag = torch.reciprocal(torch.sqrt(D))
        A_wave = diag.view((-1, 1)) * A * diag.view((1, -1))
        L = I - A_wave
        L_hat = (2.0 * L) / lambda_max - I
        return L_hat

    def forward(self, X, A, lambda_max=None):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :param A: Input adjacent matrix.
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        lambda_max = 2.0 if lambda_max is None else lambda_max
        L_hat = self.norm(A, lambda_max)
        Tx_0 = X
        out = torch.matmul(Tx_0, self.weight[0])
        
        if self.weight.size(0) > 1:
            Tx_1 = torch.einsum("ij,jkl->kil", [L_hat, X.permute(1, 0, 2)])
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * torch.einsum("ij,jkl->kil", [L_hat, Tx_1.permute(1, 0, 2)]) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out


class SAGEConv(nn.Module):
    """
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    """
    def __init__(self, in_channels, out_channels, 
                concat=True, normalize=False, bias=True):
        """
        :param in_channels: Number of input features at each node.
        :param out_channels: Desired number of output channels at each node.
        :param concat: Choose to concatenate current node features with aggregated ones.
        :param normalize: Out features will be l2-normalization.
        :param bias: Layer will learn an additive bias.
        """
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels if not concat else 2 * in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.shape[0])
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, X, A, add_self_loop=True):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :param A: Input adjacent matrix.
        :param add_self_loop: Add self-loop but if concat is True this will be ignored.
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        sz = X.shape
        adj = A.clone()
        if not self.concat and add_self_loop:
            idx = torch.arange(sz[1], device=X.device)
            adj[idx, idx] = 1
        
        out = torch.matmul(adj, X)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)

        if self.concat:
            out = torch.cat([X, out], dim=-1)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out


class GATConv(nn.Module):
    """
    The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    """
    def __init__(self, in_channels, out_channels, dropout=0):
        """
        :param in_channels: Number of input features at each node.
        :param out_channels: Desired number of output channels at each node.
        :param dropout: Dropout probability of the normalized attention coefficients.
        """
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.alpha = nn.Parameter(torch.Tensor(2 * out_channels, 1))

        self.query = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.key = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.alpha.data, gain=1.414)
        nn.init.xavier_uniform_(self.query.data, gain=1.414)
        nn.init.xavier_uniform_(self.key.data, gain=1.414)
    
    def forward(self, X, A, add_self_loop=True):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :param A: Input adjacent matrix.
        :param add_self_loop: Add self-loop but if concat is True this will be ignored.
        :param adj_available: Multiply out with adjacent matrix if sets True.
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        B, N, _ = X.shape
        adj = A.clone()
        if add_self_loop:
            idx = torch.arange(N, device=X.device)
            adj[idx, idx] = 1
        
        # map X to shape [B, N, out_channels]
        out = torch.matmul(X, self.weight)
        query = torch.matmul(X, self.query)
        key = torch.matmul(X, self.key)

        # calculate attention matrix of shape [B, N, N]

        # method 1
        # att_left = torch.matmul(out, self.alpha[:self.out_channels])
        # att_right = torch.matmul(out, self.alpha[self.out_channels:])
        # att_vec = F.leaky_relu(torch.bmm(att_left, att_right.permute(0, 2, 1)))

        # method 2
        # att_left = torch.matmul(out, self.alpha[:self.out_channels])
        # att_right = torch.matmul(out, self.alpha[self.out_channels:])
        # att_left = att_left.repeat(1, 1, N).view(B, N*N, 1)
        # att_right = att_right.repeat(1, N, 1)
        # att_vec = F.leaky_relu(att_left + att_right).view(B, N, N)

        # method 3
        # att_input = torch.bmm(out, out.permute(0, 2, 1))

        # method 4
        # att_input = torch.bmm(out, query.permute(0, 2, 1))

        # method 5
        att_input = torch.bmm(query, key.permute(0, 2, 1))
        att_vec = F.leaky_relu(att_input)

        zero_vec = -9e15*torch.ones_like(att_vec)
        attention = torch.where(adj > 0, att_vec, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        out = torch.matmul(attention, out)

        return out