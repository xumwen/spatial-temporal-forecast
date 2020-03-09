import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2


class MLELoss(nn.Module):
    def __init__(self):
        super(MLELoss, self).__init__()

    def forward(self, input, target):
        sz = input.size()
        input = input.view(sz[0], sz[1], sz[2] // 2, 2)

        mean = input[:, :, :, 0]
        log_std = input[:, :, :, 1]

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)

        logprob = normal.log_prob(target)

        return -logprob.mean()

def inferrence(prob, infer_type='mean'):
    sz = prob.size()
    prob = prob.view(sz[0], sz[1], sz[2] // 2, 2)

    mean = prob[:, :, :, 0]
    log_std = prob[:, :, :, 1]

    log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
    std = log_std.exp()

    if infer_type == 'mean':
        return mean
    elif infer_type == 'sample':
        normal = Normal(mean, std)
        # sample for 1000 times
        samples = normal.sample([1000])
        samples = torch.median(samples, dim=0)[0]
        print(samples.size())
        return samples





