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


class Decoder(nn.Module):
    def __init__(self, num_features, num_timesteps_output, hidden_size, overlap_size, use_pos_encode):
        super(Decoder, self).__init__()

        self.overlap_size = overlap_size
        self.use_pos_encode = use_pos_encode

        self.num_timesteps_output = num_timesteps_output

        self.register_buffer('position', torch.arange(num_timesteps_output))
        self.pos_encode = nn.Embedding(num_timesteps_output, 4)

        rnn_input_size = overlap_size + 4 * int(use_pos_encode)

        self.rnn_cell = nn.GRUCell(rnn_input_size, hidden_size)
        self.linear = nn.Linear(hidden_size * 2, 1)

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

            if self.use_pos_encode:
                decode_last = torch.cat([last, pos_encode[:, step, :]], dim=-1)
            else:
                decode_last = last

            hidden = self.rnn_cell(decode_last, hidden)

            out = self.linear(torch.cat([hidden, context], dim=-1))
            decoder_out.append(out)

            # roll last value
            last = torch.cat(
                [out.detach(), last[:, :self.overlap_size - 1]], dim=-1
            )

        decoder_out = torch.cat(decoder_out, dim=-1)
        return decoder_out


class Seq2Seq(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, hidden_size, overlap_size, use_pos_encode):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(
            num_features, num_timesteps_input, hidden_size, overlap_size, use_pos_encode)
        self.decoder = Decoder(
            num_features, num_timesteps_output, hidden_size, overlap_size, use_pos_encode)

    def forward(self, X):
        '''
        :param: X of shape (batch_size, num_timesteps_input, num_features)
        '''
        encoder_out, encoder_hid, last = self.encoder(X)

        encoder_hid = encoder_hid.squeeze(dim=0)

        decoder_out = self.decoder(encoder_out, encoder_hid, last)

        return decoder_out


class KSeq2Seq(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, hidden_size, overlap_size, use_pos_encode, parallel):
        super(KSeq2Seq, self).__init__()

        self.num_nodes = num_nodes
        self.num_timesteps_output = num_timesteps_output

        self.module_list = nn.ModuleList()
        for rep in range(parallel):
            self.module_list.append(
                Seq2Seq(num_nodes, num_features, num_timesteps_input,
                        num_timesteps_output, hidden_size, overlap_size, use_pos_encode)
            )

        self.attn = nn.Parameter(torch.FloatTensor(num_nodes, parallel))
        self.attn.data.normal_()

    def forward(self, X):
        # reshape to (batch_size * num_nodes, num_timesteps_input, num_features)
        inputs = X.view(-1, X.size(2), X.size(3))

        outs = []

        for idx in range(len(self.module_list)):
            out = self.module_list[idx](inputs)
            out = out.view(-1, self.num_nodes, self.num_timesteps_output)

            outs.append(out.unsqueeze(dim=-1))

        outs = torch.cat(outs, dim=-1)
        attn = torch.softmax(self.attn, dim=-1)

        outs = torch.einsum('ijkl,jl->ijk', outs, attn)

        return outs


class KRNN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64, overlap_size=1, use_pos_encode=False, parallel=10):
        super(KRNN, self).__init__()

        self.seq2seq = KSeq2Seq(num_nodes, num_features, num_timesteps_input,
                                num_timesteps_output, hidden_size, overlap_size, use_pos_encode, parallel)

    def forward(self, A, X):
        return self.seq2seq(X)
