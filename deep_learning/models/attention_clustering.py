import torch
import torch.nn as nn

from .layers import Convx2

class AttentionClustering(nn.Module):
    def __init__(self, input_channels, output_channels, query_dim, n_clusters):
        super().__init__()
        self.query_encoder = nn.Sequential(
                Convx2(input_channels, query_dim, norm=None, padding_mode='replicate'),
                nn.Conv2d(query_dim, query_dim, 1)
        )

        self.cluster_mu = nn.Parameter(torch.randn(1, n_clusters, query_dim, 1, 1))
        # self.cluster_lv = nn.Parameter(torch.zeros(1, n_clusters, 1, 1))
        # self.cluster_prior = nn.Parameter(torch.zeros(1, n_clusters, 1, 1))
        self.cluster_label = nn.Parameter(torch.randn(output_channels, n_clusters))

    def forward(self, x):
        query = self.query_encoder(x).unsqueeze(1)
        # logit = self.cluster_prior - torch.sum(torch.pow(query - self.cluster_mu, 2), dim=2) / torch.exp(self.cluster_lv)
        logit = -torch.sum(torch.pow(query - self.cluster_mu, 2), dim=2)

        attention = torch.softmax(logit, dim=1) # BxNCxHxW
        B, NC, H, W = attention.shape
        attention = attention.view(B, NC, H*W)

        res = torch.matmul(self.cluster_label, attention)
        res = res.view(B, -1, H, W)

        return res
