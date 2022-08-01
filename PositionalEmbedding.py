import torch
import torch.nn as nn


## code from https://github.com/gusdnd852

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model, device):
        super(PositionalEmbedding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        _t = torch.arange(0, max_len, device=device)
        _t = _t.float().unsqueeze(dim=1)
        _i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(_t / 10000 ** (_i / d_model))
        self.encoding[:, 1::2] = torch.cos(_t / 10000 ** (_i / d_model))

    def forward(self, x):
        ## input shape: [batch_size(N), length(L), embed_size(d_model)]
        N, L = x.size()

        return self.encoding[:L, :]
