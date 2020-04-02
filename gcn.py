import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


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
        A = A.clone()
        if not self.concat and add_self_loop:
            idx = torch.arange(sz[1], device=X.device)
            A[idx, idx] = 1
        
        out = torch.matmul(A, X)
        out = out / A.sum(dim=-1, keepdim=True).clamp(min=1)

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
        self.a = nn.Parameter(torch.zeros(size=(2 * out_channels, 1)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, X, A, add_self_loop=True, adj_available=False):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :param A: Input adjacent matrix.
        :param add_self_loop: Add self-loop but if concat is True this will be ignored.
        :param adj_available: Multiply out with adjacent matrix if sets True.
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        B, N, _ = X.shape
        A = A.clone()
        if add_self_loop:
            idx = torch.arange(N, device=X.device)
            A[idx, idx] = 1
        
        # map X to shape [B, N, out_channels]
        X = torch.matmul(X, self.weight)

        # calculate attention matrix of shape [B, N, N]
        att_left = X.repeat(1, 1, N).view(B, N*N, -1)
        att_right = X.repeat(1, N, 1)
        att_input = torch.cat([att_left, att_right], dim=-1)
        att_vec = F.leaky_relu(torch.matmul(att_input, self.a).squeeze(-1)).view(B, N, N)

        zero_vec = -9e15*torch.ones_like(att_vec)
        attention = torch.where(A > 0, att_vec, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        out = torch.matmul(attention, X)

        if adj_available:
            out = torch.matmul(A, out)

        return out


class PyGConv(nn.Module):
    """
    Choose GCN implemented by pytorch-geometric and apply to a batch of nodes.
    """
    def __init__(self, in_channels, out_channels, gcn_type, gcn_partition=None):
        """
        :param in_channels: Number of input features at each node.
        :param out_channels: Desired number of output channels at each node.
        :param gcn_type: Choose GCN type.
        :param gcn_partition: Choose GCN partition method.
        """
        super(PyGConv, self).__init__()

        self.out_channels = out_channels

        # Use edge_weight argument in forward
        self.adj_available = True
        # Use node_dim argument for batch training
        self.batch_training = False
        # Use partition to train on mini-batch of sub-graph
        self.gcn_partition = gcn_partition
        self.kwargs = {'in_channels':in_channels, 'out_channels':out_channels}

        if self.gcn_partition == 'cluster':
            self.gcn = PyGConv(in_channels, out_channels, gcn_type, gcn_partition=None)
        elif self.gcn_partition == 'sample':
            # Sampled edge are usually unsymmetric so only support spatial domain gcn
            assert gcn_type in ['sage', 'graph', 'gat']
            self.gcn1 = PyGConv(in_channels, out_channels, gcn_type, gcn_partition=None)
            self.gcn2 = PyGConv(out_channels, out_channels, gcn_type, gcn_partition=None)
        else:
            if gcn_type == 'gat':
                self.adj_available = False
            if gcn_type in ['normal', 'cheb', 'graph', 'sage']:
                self.batch_training = True
                self.kwargs['node_dim'] = 1
            if gcn_type == 'cheb':
                self.kwargs['K'] = 3
            if gcn_type == 'sage':
                self.kwargs['concat'] = True
            
            GCNCell = {'normal':PyG.GCNConv, 
                        'cheb':PyG.ChebConv,
                        'sage':PyG.SAGEConv, 
                        'graph':PyG.GraphConv,
                        'gat':PyG.GATConv}\
                        .get(gcn_type)
            
            self.gcn = GCNCell(**self.kwargs)
    
    def get_batch(self, X):
        # Wrap input node and edge features, along with the single edge_index, into a `torch_geometric.data.Batch` instance
        data_list = [Data(x=x) for x in X]

        return Batch.from_data_list(data_list)

    def forward(self, X, edge_index, edge_weight):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :param edge_index: Graph connectivity in COO format with shape(2, num_edges)
        :param edge_weight: Edge feature matrix with shape (num_edges, num_edge_features)
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        sz = X.shape
        if self.gcn_partition == 'cluster':
            out = torch.zeros(sz[0], sz[1], self.out_channels, device=X.device)
            graph_data = Data(edge_index=edge_index, edge_attr=edge_weight, 
                                train_mask=torch.arange(0, sz[1]), num_nodes=sz[1])
            cluster_data = ClusterData(graph_data, num_parts=50, recursive=False, save_dir='./data')
            loader = ClusterLoader(cluster_data, batch_size=5, shuffle=True, num_workers=0)

            for subgraph in loader:
                out[:, subgraph.train_mask] = self.gcn(X[:, subgraph.train_mask], 
                                                subgraph.edge_index.to(X.device), 
                                                subgraph.edge_attr.to(X.device))

        elif self.gcn_partition == 'sample':
            # Use NeighborSampler() to iterates over graph nodes in a mini-batch fashion 
            # and constructs sampled subgraphs (use cpu for no CUDA version)
            out = torch.zeros(sz[0], sz[1], self.out_channels, device=X.device)
            graph_data = Data(edge_index=edge_index, num_nodes=sz[1]).to('cpu')
            loader = NeighborSampler(graph_data, size=[10, 5], num_hops=2, batch_size=100,
                         shuffle=True, add_self_loops=False)

            for data_flow in loader():
                block1 = data_flow[0]
                t = self.gcn1(X, edge_index[:, block1.e_id], edge_weight[block1.e_id])
                block2 = data_flow[1]
                part_out = self.gcn2(t, edge_index[:, block2.e_id], edge_weight[block2.e_id])
                out[:, data_flow.n_id] = part_out[:, data_flow.n_id]

        elif self.batch_training:
            if self.adj_available:
                out = self.gcn(X, edge_index, edge_weight)
            else:
                out = self.gcn(X, edge_index)

        else:
            # Currently, conv in [SAGEConv, GATConv] cannot use argument node_dim for batch training
            # This is a temp solution but it's very very very slow!
            # Costing about 6 times more than batch_training
            batch = self.get_batch(X)
            if self.adj_available:
                out = self.gcn(batch.x, edge_index, edge_weight)
            else:
                out = self.gcn(batch.x, edge_index)
            out = out.view(sz[0], sz[1], -1)
        
        return out


class GCNUnit(nn.Module):
    """
    Choose GCNUnit with package and type.
    """
    def __init__(self, in_channels, out_channels, gcn_type, gcn_package, gcn_partition=None):
        """
        :param in_channels: Number of input features at each node.
        :param out_channels: Desired number of output channels at each node.
        :param gcn_type: Choose GCN type.
        :param gcn_package: Choose GCN package in ['pyg', 'ours'].
        :param gcn_partition: Choose GCN partition method in ['cluster', 'sample']
        """
        super(GCNUnit, self).__init__()
        self.adj_type = 'sparse'
        if gcn_package == 'pyg':
            self.gcn = PyGConv(in_channels=in_channels,
                                out_channels=out_channels,
                                gcn_type=gcn_type,
                                gcn_partition=gcn_partition)
        else:
            assert gcn_type in ['normal', 'cheb', 'sage', 'gat']
            self.adj_type = 'dense'
            GCNCell = {'normal':GCNConv, 
                        'cheb':ChebConv, 
                        'sage':SAGEConv, 
                        'gat':GATConv}.get(gcn_type)
            self.gcn = GCNCell(in_channels=in_channels,
                                out_channels=out_channels)

    def forward(self, X, A=None, edge_index=None, edge_weight=None):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :param **kwargs: Additional arguments(dense or sparse adj matrix).
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        if self.adj_type == 'sparse':
            out = self.gcn(X, edge_index=edge_index, edge_weight=edge_weight)
        else:
            out = self.gcn(X, A=A)
        
        return out
