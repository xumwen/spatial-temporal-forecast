import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.Theta = nn.Parameter(torch.FloatTensor(in_channels,
                                                    out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta.shape[1])
        self.Theta.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, in_channels)
        :return: Output data of shape (batch_size, num_nodes, out_channels)
        """
        t = torch.einsum("ij,jkl->kil", [A_hat, X.permute(1, 0, 2)])
        output = torch.matmul(t, self.Theta)
        return output