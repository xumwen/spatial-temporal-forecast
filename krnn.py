import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class KRNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64, num_comps=10):
        """
        build one RNN for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(KRNN, self).__init__()
        self.grus = nn.ModuleList()
        self.linears = nn.ModuleList()

        for r in range(num_comps):
            self.grus.append(
                nn.GRU(num_features, hidden_size)
            )
            self.linears.append(
                nn.Linear(hidden_size, num_timesteps_output)
            )

        self.embed = nn.Parameter(torch.FloatTensor(num_nodes, num_comps))
        self.embed.data.normal_()


    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """

        out = []

        sz = X.size()
        X = X.view(-1, sz[2], sz[3]).permute(1, 0, 2)

        for i in range(len(self.grus)):
            hid, _ = self.grus[i](X)
            hid = hid.mean(dim=0)
            hid = hid.view(sz[0], sz[1], -1)
            out.append(
                self.linears[i](hid).unsqueeze(dim=-1)
            )
        
        out = torch.cat(out, dim=-1)
        weight = torch.softmax(self.embed, dim=-1)

        out = torch.einsum('ijkl,jl->ijk', out, weight)

        return out
