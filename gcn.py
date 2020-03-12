import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.data import Data, Batch, DataLoader

class GCNConv(nn.Module):
    """
    Neural network block that applies a graph convolution to a batch of nodes.
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
        I = torch.ones(A.shape[0]).to(A.device)
        # Remove self-loops
        if A[0][0] != 0:
            A = A - torch.diag(A[0][0] * I)
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


class PyGConv(nn.Module):
    """
    Applies a graph convolution implemented by pytorch geometric to a batch of nodes.
    """
    def __init__(self, in_channels, out_channels, gcn_type):
        """
        :param in_channels: Number of input features at each node.
        :param out_channels: Desired number of output channels at each node.
        """
        super(PyGConv, self).__init__()

        # Use edge_weight argument in forward
        self.adj_available = True
        self.batch_training = True
        self.kwargs = {'in_channels':in_channels, 'out_channels':out_channels}

        if gcn_type == 'gat':
            self.adj_available = False
        if gcn_type not in ['normal', 'cheb', 'graph']:
            self.batch_training = False
        if gcn_type == 'cheb':
            self.kwargs['K'] = 3
        if gcn_type in ['normal', 'cheb', 'graph']:
            self.kwargs['node_dim'] = 1
        
        GCNCell = {'normal':PyG.GCNConv, 
                    'cheb':PyG.ChebConv,
                    'sage':PyG.SAGEConv, 
                    'graph':PyG.GraphConv,
                    'gat':PyG.GATConv}\
                    .get(gcn_type)
        
        self.gcn = GCNCell(**self.kwargs)
    
    def get_batch(self, X):
        # Wrap input node and edge features, along with the single edge_index, into a `torch_geometric.data.Batch` instance
        batch_size = X.shape[0]
        data_list = [Data(x=x) for x in X]

        return Batch.from_data_list(data_list)

    def forward(self, X, edge_index, edge_weight):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :param edge_index: Graph connectivity in COO format with shape(2, num_edges)
        :param edge_weight: Edge feature matrix with shape (num_edges, num_edge_features)
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        if self.batch_training:
            if self.adj_available:
                out = self.gcn(X, edge_index, edge_weight)
            else:
                out = self.gcn(X, edge_index)
        else:
            # Currently, several conv cannot use argument node_dim to batch training
            # This is a temp solution but it's very very very slow!
            batch = self.get_batch(X)
            if self.adj_available:
                out = self.gcn(batch.x, edge_index, edge_weight)
            else:
                out = self.gcn(batch.x, edge_index)
        
        return out.view(X.shape[0], X.shape[1], -1)