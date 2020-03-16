import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np


class Encoder(nn.Module):
    def __init__(self, num_features, num_timesteps_input, hidden_size, overlap_size, use_pos_encode):
        super(Encoder, self).__init__()

        self.num_features = num_features
        self.overlap_size = overlap_size
        self.use_pos_encode = use_pos_encode

        self.register_buffer('position', torch.arange(num_timesteps_input))
        self.pos_encode = nn.Embedding(num_timesteps_input, 4)

        rnn_input_size = num_features * overlap_size + \
            4 * int(use_pos_encode)

        self.rnn = nn.GRU(rnn_input_size, hidden_size)

    def get_overlap_inputs(self, inputs, overlap_size):
        overlap_inputs = []
        for rep in range(overlap_size):
            shift_inputs = inputs.roll(rep, dims=1)
            # pad sequence with 0
            shift_inputs[:, :rep, :] = 0
            overlap_inputs.append(shift_inputs)
        return torch.cat(overlap_inputs, dim=2)

    def add_position_encode(self, inputs):
        pos_encode = self.pos_encode(self.position)
        pos_encode = pos_encode.expand(
            (inputs.size(0), ) + pos_encode.size()
        )

        inputs = torch.cat([inputs, pos_encode], dim=-1)
        return inputs

    def forward(self, inputs):
        """
        :param inputs: Input data of shape (batch_size, num_nodes, num_timesteps, num_features).
        """
        inputs = self.get_overlap_inputs(
            inputs, overlap_size=self.overlap_size)

        if self.use_pos_encode:
            encode_inputs = self.add_position_encode(inputs)
        else:
            encode_inputs = inputs

        inputs = inputs.permute(1, 0, 2)
        encode_inputs = encode_inputs.permute(1, 0, 2)

        out, hidden = self.rnn(encode_inputs)

        # extract last input of encoder, used for decoder
        # 0 indicates target dim
        last = inputs.view(inputs.size(0), inputs.size(1),
                           self.overlap_size, self.num_features
                           )[-1, :, :, 0]

        return out, hidden, last.detach()


class KEncoder(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, hidden_size, overlap_size, use_pos_encode, parallel):
        super(KEncoder, self).__init__()

        self.num_nodes = num_nodes

        self.module_list = nn.ModuleList()
        for rep in range(parallel):
            self.module_list.append(
                Encoder(num_features, num_timesteps_input,
                        hidden_size, overlap_size, use_pos_encode)
            )

        self.attn = nn.Parameter(torch.FloatTensor(num_nodes, parallel))
        self.attn.data.normal_()

    def forward(self, X):
        # reshape to (batch_size * num_nodes, num_timesteps_input, num_features)
        inputs = X.reshape(-1, X.size(2), X.size(3))

        hiddens = []

        for idx in range(len(self.module_list)):
            out, hidden, _ = self.module_list[idx](inputs)
            hiddens.append(
                hidden.view(X.size(0), X.size(1), -1).unsqueeze(dim=-1)
            )

        hiddens = torch.cat(hiddens, dim=-1)
        hiddens = torch.einsum('ijkl,jl->ijk', hiddens, self.attn)
        return hiddens


class KRNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, hidden_size=64, overlap_size=3, use_pos_encode=False, parallel=10):
        super(KRNN, self).__init__()

        self.kencoder = KEncoder(num_nodes, num_features, num_timesteps_input,
                                 hidden_size, overlap_size, use_pos_encode, parallel)

    def forward(self, X):
        return self.kencoder(X)
