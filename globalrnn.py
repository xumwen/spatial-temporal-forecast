import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class GlobalRNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64):
        """
        build one RNN for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(GlobalRNN, self).__init__()
        self.gru = nn.GRU(num_features, hidden_size)
        self.linears = nn.ModuleList()

        for n in range(num_nodes):
            self.linears.append(
                nn.Linear(hidden_size, num_timesteps_output)
            )

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """
        sz = X.size()
        X = X.view(-1, sz[2], sz[3]).permute(1, 0, 2)
        hid, _ = self.gru(X)
        hid = hid.mean(dim=0)
        hid = hid.view(sz[0], sz[1], -1)

        out = []
        for n in range(hid.size(1)):
            out.append(
                self.linears[n](hid[:, n, :]).unsqueeze(dim=1)
            )
        out = torch.cat(out, dim=1)

        return out
