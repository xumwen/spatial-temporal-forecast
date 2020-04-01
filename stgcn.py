import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GCNUnit
from my_conv import Conv2dSame as Conv2d

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=[2,3,4,5]):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv_list = nn.ModuleList()
        for _kernel_size in kernel_size:
            self.conv_list.append(Conv2d(in_channels, out_channels, (1, _kernel_size)))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        for i in range(len(self.conv_list)):
            if i == 0:
                out = self.conv_list[i](X)
            else:
                out += self.conv_list[i](X)
        out = F.relu(out)
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                num_nodes, gcn_type, gcn_package, gcn_partition):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.gcn = GCNUnit(in_channels=out_channels,
                            out_channels=spatial_channels,
                            gcn_type=gcn_type,
                            gcn_package=gcn_package,
                            gcn_partition=gcn_partition)
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, A, edge_index, edge_weight):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A: Adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes, num_timesteps_out,
        num_features=out_channels).
        """
        t1 = self.temporal1(X)
        # batch_size * timesteps -> batch_size
        t21 = t1.permute(0, 2, 1, 3).contiguous().view(-1, t1.shape[1], t1.shape[3])
        t22 = F.relu(self.gcn(t21, A, edge_index, edge_weight))
        # batch_size -> (batch_size, timesteps)
        t23 = t22.view(t1.shape[0], t1.shape[2], t22.shape[1], t22.shape[2]).permute(0, 2, 1, 3)
        t3 = self.temporal2(t23)
        
        return self.batch_norm(t3)


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_edges, num_features, 
                num_timesteps_input, num_timesteps_output, 
                gcn_type='cheb', gcn_package='pyg',
                gcn_partition=None, **kwargs):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes,
                                 gcn_type=gcn_type, gcn_package=gcn_package,
                                 gcn_partition=gcn_partition)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes,
                                 gcn_type=gcn_type, gcn_package=gcn_package,
                                 gcn_partition=gcn_partition)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear(num_timesteps_input * 64,
                               num_timesteps_output)

    def forward(self, X, A=None, edge_index=None, edge_weight=None):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A, edge_index, edge_weight)
        out2 = self.block2(out1, A, edge_index, edge_weight)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4
