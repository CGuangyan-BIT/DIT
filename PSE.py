#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from token_transformer import Token_transformer
import torch.nn.functional as F

def knn(x, k):
    x = x.transpose(1,2)
    distance = -(torch.sum((x.unsqueeze(1) - x.unsqueeze(2)).pow(2), -1) + 1e-7)
    idx = distance.topk(k=k, dim=-1)[1]
    return idx


def get_local_feature(x, refer_idx):

    x = x.view(*x.size()[:3])

    batch_size, num_points, k = refer_idx.size()

    idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1) * num_points

    idx = refer_idx + idx_base

    idx = idx.view(-1)

    _, _, num_dims = x.size()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k*num_dims)

    return feature

class LFI(nn.Module):
    def __init__(self):
        super(LFI,self).__init__()
    def forward(self,x,refer_idx):
        x = get_local_feature(x,refer_idx)
        return x

class PSE_module(nn.Module):
    def __init__(self, num_points=20, in_chans=3, embed_dim=256, token_dim=64,norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_points = num_points
        self.LFI0 = LFI()
        self.LFI1 = LFI()
        self.LFI2 = LFI()
        self.LFI3 = LFI()
        self.LFI4 = LFI()
        self.LFI5 = LFI()
        self.LFI6 = LFI()
        self.LFI7 = LFI()

        self.attention1 = Token_transformer(dim=in_chans * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention2 = Token_transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention3 = Token_transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention4 = Token_transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention5 = Token_transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention6 = Token_transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention7 = Token_transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        self.attention8 = Token_transformer(dim=token_dim * 20, in_dim=token_dim, num_heads=4, mlp_ratio=1.0)
        
        self.project = nn.Linear(token_dim * 4, embed_dim)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        refer_idx = knn(x,self.num_points)
        refer_idx = refer_idx.to('cuda')

        x = x.transpose(1,2).contiguous()
        x = self.LFI0(x,refer_idx)
        x1 = self.attention1(x)
        
        x = self.LFI1(x1,refer_idx)
        x2 = self.attention2(x)

        x = self.LFI2(x2,refer_idx)
        x3 = self.attention3(x)

        x = self.LFI3(x3,refer_idx)
        x4 = self.attention4(x)

        x = self.LFI4(x4,refer_idx)
        x5 = self.attention5(x)
        
        x = self.LFI5(x5,refer_idx)
        x6 = self.attention6(x)

        x = self.LFI6(x6,refer_idx)
        x7 = self.attention7(x)

        x = self.LFI7(x7,refer_idx)
        x8 = self.attention8(x)

        x = F.leaky_relu(torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=-1))

        x = self.norm(x)
        x = x.transpose(-1,-2).contiguous()
        return x
