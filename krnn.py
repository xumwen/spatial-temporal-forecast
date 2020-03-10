import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np


class Encoder(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, hidden_size, overlap_size, use_pos_encode):
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

        inputs = inputs.permute(1, 0, 2)
        encode_inputs = encode_inputs.permute(1, 0, 2)

        out, hidden = self.rnn(encode_inputs)

        last = inputs.view(inputs.size(0), inputs.size(1),
                           self.overlap_size, self.num_features
                           )[-1, :, :, 0]

        return out, hidden, last


class KEncoder(nn.Module):
    def __init__(self, net_cls, parallel, **kwargs):
        super(KEncoder, self).__init__()
        self.module_list = nn.ModuleList()
        for idx in range(parallel):
            self.module_list.append(net_cls(**kwargs))

        num_nodes = kwargs['num_nodes']

        self.attn = nn.Parameter(torch.FloatTensor(num_nodes, parallel))
        self.attn.data.normal_()

    def forward(self, X):
        inputs = X.view(-1, X.size(2), X.size(3))

        out_lst, hidden_lst = [], []
        for idx in range(len(self.module_list)):
            out, hidden, last = self.module_list[idx](inputs)

            out = out.view(out.size(0), X.size(0), X.size(1), out.size(2))
            hidden = hidden.view(hidden.size(0), X.size(0),
                                 X.size(1), hidden.size(2))
            last = last.view(X.size(0), X.size(1), last.size(1))

            out_lst.append(out.unsqueeze(dim=-1))
            hidden_lst.append(hidden.unsqueeze(dim=-1))

        out_lst = torch.cat(out_lst, dim=-1)
        hidden_lst = torch.cat(hidden_lst, dim=-1)

        attn = torch.softmax(self.attn, dim=-1)

        out = torch.einsum('ijklm,km->ijkl', out_lst, attn)
        hidden = torch.einsum('ijklm,km->ijkl', hidden_lst, attn)

        return out, hidden, last


class Decoder(nn.Module):
    def __init__(self, num_timesteps_output, num_features, overlap_size, use_pos_encode, hidden_size=64):
        super(Decoder, self).__init__()

        self.overlap_size = overlap_size
        self.use_pos_encode = use_pos_encode

        self.num_timesteps_output = num_timesteps_output

        self.register_buffer('position', torch.arange(num_timesteps_output))
        self.pos_encode = nn.Embedding(num_timesteps_output, 4)

        rnn_input_size = overlap_size + \
            4 * int(use_pos_encode)

        self.rnn_cell = nn.GRUCell(rnn_input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, encoder_out, encoder_hid, last):
        '''
        :param encoder_out: (num_timesteps_input, batch_size, hidden_size)
        :param encoder_hid: (batch_size, hidden_size)
        :param last: shape (batch_size, overlap_size)
        '''

        decoder_out = []

        hidden = encoder_hid

        pos_encode = self.pos_encode(self.position)
        pos_encode = pos_encode.expand(
            (encoder_hid.size(0), ) + pos_encode.size()
        )

        for step in range(self.num_timesteps_output):
            attn_w = torch.einsum('ijk,jk->ij', encoder_out, hidden)
            attn_w = F.softmax(attn_w, dim=0)
            context = torch.einsum('ijk,ij->jk', encoder_out, attn_w)

            hidden = hidden + context
            encode_last = torch.cat([last, pos_encode[:, step, :]], dim=-1)

            hidden = self.rnn_cell(encode_last, hidden)

            out = self.linear(hidden)
            decoder_out.append(out)

            last.data[:, 0] = out.view(-1, ).data

        decoder_out = torch.cat(decoder_out, dim=-1)
        return decoder_out


class KRNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64, overlap_size=1, use_pos_encode=True, parallel=10):
        super(KRNN, self).__init__()

        self.encoder = KEncoder(net_cls=Encoder, parallel=parallel,
                                num_nodes=num_nodes, num_features=num_features, num_timesteps_input=num_timesteps_input, hidden_size=hidden_size,
                                overlap_size=overlap_size, use_pos_encode=use_pos_encode
                                )

        self.decoder = Decoder(num_timesteps_output, num_features,
                               overlap_size, use_pos_encode, hidden_size=hidden_size)

    def forward(self, A, X):
        # shape of out (num_tmiesteps_input, batch_size, num_nodes, hidden_size)
        # shape of hidden (1, batch_size, num_nodes, hidden_size)

        batch_size, num_nodes = X.size(0), X.size(1)

        encoder_out, encoder_hid, last = self.encoder(X)

        encoder_out = encoder_out.reshape(
            encoder_out.size(0), -1, encoder_out.size(-1))
        encoder_hid = encoder_hid.reshape(-1, encoder_hid.size(-1))
        last = last.reshape(-1, last.size(-1))

        decoder_out = self.decoder(encoder_out, encoder_hid, last)
        decoder_out = decoder_out.view(batch_size, num_nodes, -1)
        return decoder_out
