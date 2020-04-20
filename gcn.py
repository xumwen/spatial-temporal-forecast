import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader

import dense_gcn as ours
from dense_egnn import EGNN, SAGELA
from sparse_egnn import SAGELANet, GatedGCNNet, MyEGNNNet


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
            assert gcn_type not in ['cheb', 'normal']
            self.gcn1 = PyGConv(in_channels, out_channels, gcn_type, gcn_partition=None)
            self.gcn2 = PyGConv(out_channels, out_channels, gcn_type, gcn_partition=None)
        else:
            if gcn_type == 'gat':
                self.adj_available = False
            if gcn_type not in ['gat']:
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
                        'gat':PyG.GATConv,
                        'sagela':SAGELANet,
                        'gated':GatedGCNNet,
                        'my':MyEGNNNet}\
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
        if torch.is_tensor(X):
            sz = X.shape
        if self.gcn_partition == 'cluster':
            out = torch.zeros(sz[0], sz[1], self.out_channels, device=X.device)
            graph_data = Data(edge_index=edge_index, edge_attr=edge_weight, 
                                train_mask=torch.arange(0, sz[1]), num_nodes=sz[1]).to('cpu')
            cluster_data = ClusterData(graph_data, num_parts=50, recursive=False, save_dir='./data/cluster')
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
            # Currently, conv in [GATConv] cannot use argument node_dim for batch training
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
            assert gcn_type in ['normal', 'cheb', 'sage', 'gat', 'egnn', 'sagela']
            self.adj_type = 'dense'
            GCNCell = {'normal':ours.GCNConv, 
                        'cheb':ours.ChebConv, 
                        'sage':ours.SAGEConv, 
                        'gat':ours.GATConv,
                        'egnn':EGNN,
                        'sagela':SAGELA}.get(gcn_type)
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
