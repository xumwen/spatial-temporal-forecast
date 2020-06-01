import torch
from torch.nn import Parameter, Module, init

class ValueNorm(Module):
    def __init__(self, num_nodes, eps=1e-5, momentum=0.1, 
                affine=True, track_running_stats=True):
        super(ValueNorm, self).__init__()
        self.num_nodes = num_nodes
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_nodes))
            self.bias = Parameter(torch.Tensor(num_nodes))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_nodes))
            self.register_buffer('running_var', torch.ones(num_nodes))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    
    def forward(self, input, indices):
        # use exponential moving average
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  
                    exponential_average_factor = self.momentum
       
        mean = input.mean(dim=[1, 2])
        var = input.var(dim=[1, 2])
        # update running statics
        if self.training and self.track_running_stats:
            self.running_mean[indices].data = ((1.0 - exponential_average_factor) * self.running_mean[indices] + \
                exponential_average_factor * mean).data
            self.running_var[indices].data = ((1.0 - exponential_average_factor) * self.running_var[indices] + \
                exponential_average_factor * var).data
        
        # normalize
        if self.track_running_stats:
            mean = self.running_mean[indices]
            var = self.running_var[indices]
        output = (input - mean.view(-1, 1, 1)) / (var.view(-1, 1, 1) + self.eps).sqrt()
        if self.affine:
            output = self.weight[indices].view(-1, 1, 1) * output + self.bias[indices].view(-1, 1, 1)

        return output
