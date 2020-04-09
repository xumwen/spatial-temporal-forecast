import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGELA(nn.Module):
    """
    The SAGE-LA operator from the `"GCN-LASE: Towards Adequately 
    Incorporating Link Attributes in Graph Convolutional Networks" 
    <https://arxiv.org/abs/1902.09817>`_ paper
    """
    def __init__(self, in_channels, out_channels, edge_feature_avalible=True):
        super(SAGELA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_avalible = edge_feature_avalible
        self.edge_channels = 3 if edge_feature_avalible else 1

        self.weight1 = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight3 = nn.Parameter(torch.Tensor(2 * out_channels, out_channels))
        self.u = nn.Parameter(torch.Tensor(self.edge_channels, in_channels))

        self.alpha_node = nn.Parameter(torch.Tensor(2 * in_channels, 1))
        self.alpha_edge = nn.Parameter(torch.Tensor(self.edge_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight2.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight3.data, gain=1.414)
        nn.init.xavier_uniform_(self.u.data, gain=1.414)
        nn.init.xavier_uniform_(self.alpha_node.data, gain=1.414)
        nn.init.xavier_uniform_(self.alpha_edge.data, gain=1.414)
    
    def forward(self, X, A, add_self_loop=True):
        B, N, _ = X.shape
        adj = A.clone()
        if add_self_loop:
            idx = torch.arange(N, device=X.device)
            adj[idx, idx] = 1
        
        # construct dge feature of shape [N, N, edge_channels]
        edge_feature = adj.unsqueeze(2)
        if self.edge_feature_avalible:
            edge_feature = torch.stack([adj, adj.t(), adj + adj.t()], dim=2)

        # calculate lambda of shape [B, N, N]
        lam_edge = torch.matmul(edge_feature, self.alpha_edge).squeeze(-1)
        lam_left = torch.matmul(X, self.alpha_node[:self.in_channels])\
            .repeat(1, 1, N).view(B, N*N, 1)
        lam_right = torch.matmul(X, self.alpha_node[self.in_channels:])\
            .repeat(1, N, 1)

        lam = (lam_left + lam_right).view(B, N, N) + lam_edge
        lam = F.sigmoid(lam + self.bias)
        zero_vec = torch.zeros_like(lam)
        lam = torch.where(edge_feature[:, :, -1] > 0, lam, zero_vec)

        # amplifier is shape [N, N, in_channels]
        amplifier = torch.matmul(edge_feature, self.u)
        aggr_out = torch.einsum('ijc,bjc->bijc', amplifier, X)
        aggr_out = torch.einsum('bij,bijc->bic', lam, aggr_out)
        aggr_out = torch.matmul(aggr_out, self.weight2)

        # concat res and aggr
        res = torch.matmul(X, self.weight1)
        out = torch.matmul(torch.cat([res, aggr_out], dim=-1), self.weight3)

        return out


class EGNN(nn.Module):
    """
    The EGNN operator from the `"Exploiting Edge 
    Features in Graph Neural Networks" 
    <https://arxiv.org/abs/1809.02709>`_ paper
    """
    def __init__(self, in_channels, out_channels, edge_feature_avalible=False):
        super(EGNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_avalible = edge_feature_avalible
        self.edge_channels = 3 if edge_feature_avalible else 1

        self.weight1 = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = nn.Parameter(torch.Tensor(self.edge_channels * out_channels, out_channels))
        self.alpha = nn.Parameter(torch.Tensor(2 * out_channels, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight2.data, gain=1.414)
        nn.init.xavier_uniform_(self.alpha.data, gain=1.414)
    
    def norm(self, E_hat):
        N, _, P = E_hat.shape
        E_wave = E_hat / E_hat.sum(axis=1).view(N, 1, P)
        return E_wave
    
    def forward(self, X, A, add_self_loop=True):
        B, N, _ = X.shape
        adj = A.clone()
        if add_self_loop:
            idx = torch.arange(N, device=X.device)
            adj[idx, idx] = 1
        
        # map X to shape [B, N, out_channels]
        out = torch.matmul(X, self.weight1)

        # construct dge feature of shape [N, N, edge_channels]
        edge_feature = adj.unsqueeze(2)
        if self.edge_feature_avalible:
            edge_feature = torch.stack([adj, adj.t(), adj + adj.t()], dim=2)
        E_wave = self.norm(edge_feature)

        # calculate attention matrix of shape [B, N, N]
        att_left = torch.matmul(out, self.alpha[:self.out_channels])
        att_right = torch.matmul(out, self.alpha[self.out_channels:])

        # method 1
        attention = F.leaky_relu(torch.bmm(att_left, att_right.permute(0, 2, 1)))
        
        # method 2
        # att_left = att_left.repeat(1, 1, N).view(B, N*N, 1)
        # att_right = att_right.repeat(1, N, 1)
        # attention = F.leaky_relu(att_left + att_right).view(B, N, N)

        # multiply attention with normalized edge feature in each edge channel
        attention = torch.einsum('bij,ijp->pbij', attention.exp(), E_wave)
        # scale
        attention = attention / attention.sum(axis=-1).view(self.edge_channels, B, N, 1)
        # use attention to aggr in each edge channel
        out = torch.einsum('pbij,bjc->bipc', attention, out)
        # concat out of all edge channels
        out = out.contiguous().view(B, N, -1)

        out = torch.matmul(out, self.weight2)

        return out