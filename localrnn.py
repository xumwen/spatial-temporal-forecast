import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRNN(nn.Module):
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
        super(LocalRNN, self).__init__()
        
        self.grus = nn.ModuleList()
        self.linears = nn.ModuleList()

        for _ in range(num_nodes):
            self.grus.append(
                nn.GRU(num_features, hidden_size)
            )
            self.linears.append(
                nn.Linear(hidden_size, num_timesteps_output)
            )

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """
        out = []
        for n in range(X.size(1)):
            input_sequence = X[:, n, :, :].permute(1, 0, 2)
            hid, _ = self.grus[n](input_sequence)
            hid = hid.mean(dim=0)
            out.append(
                self.linears[n](hid).unsqueeze(dim=1)
            )
        out = torch.cat(out, dim=1)
        return out


