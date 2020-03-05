import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class Nconv(nn.Module):
    def __init__(self):
        super(Nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class Linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(Linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class MultiGCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,num_adj=3,order=2):
        super(MultiGCN,self).__init__()
        self.nconv = Nconv()
        c_in = (order*num_adj+1)*c_in
        self.mlp = Linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,A_list):
        out = [x]
        for A in A_list:
            x1 = self.nconv(x,A)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,A)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', 
                 dropout=0.3, num_adj=2, gcn_bool=True, 
                 addaptadj=True, aptinit=None, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(GWNET, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=num_features,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1
        self.num_adj = num_adj

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.num_adj +=1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.num_adj += 1

                
                

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(MultiGCN(dilation_channels,residual_channels,dropout,num_adj=self.num_adj))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=num_timesteps_output,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def remove_self_loop(self, A):
        I = torch.ones(A.shape[0])
        if A[0][0] != 0:
            A = A - torch.diag(A[0][0] * I)
        return A

    def forward(self, A, input):
        """
        :parms: input data of shape (batch_size, num_nodes, num_timesteps_input, in_channels)
        :return: output data of shape(batch_size, num_nodes, num_timesteps_output, out_channels)
        """
        # input to (batch_size, in_channels, num_nodes, num_timesteps_input)
        input = input.permute(0, 3, 1, 2)
        
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        # A = self.remove_self_loop(A)
        A_list = [A, A.T]
        if self.gcn_bool and self.addaptadj and self.num_adj != 0:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            A_list = A_list + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.num_adj > 0:
                x = self.gconv[i](x, A_list)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # x's shape is (batch_size, num_timesteps_output, num_nodes, out_channels)
        
        return x.permute(0, 2, 1, 3).squeeze(dim=3)





