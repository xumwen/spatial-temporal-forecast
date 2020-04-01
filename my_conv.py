import torch
import math

class Conv2dSame(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv2dSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)

    def forward(self, x_in):
        N, C, H, W = x_in.shape
        H2 = math.ceil(H / self.S)
        W2 = math.ceil(W / self.S)
        Fc, Fr = self.F 
        Pr = (H2 - 1) * self.S + (Fr - 1) * self.D + 1 - H
        Pc = (W2 - 1) * self.S + (Fc - 1) * self.D + 1 - W
        x_pad = torch.nn.ZeroPad2d((Pr//2, Pr - Pr//2, Pc//2, Pc - Pc//2))(x_in)
        x_out = self.layer(x_pad)
        return x_out