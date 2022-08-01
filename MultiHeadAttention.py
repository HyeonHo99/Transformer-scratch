import torch
import torch.nn as nn
import math

# d_model: the number of expected features in the encoder/decoder inputs
# nheads: the number of heads in the multiheadattention models

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.d_head = d_model // nheads
        assert (self.nheads * self.d_head == self.d_model), "embed_size is not divisible by num_heads"

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_concat = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, q, k, v, mask):
        ## N: batch_size, L: length of sequential data
        ## q,k,v shape each : [N, L, embed_size]
        N = q.shape[0]
        L = q.shape[1]

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        ## split into multi-heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        ## scale dot product attention
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(self.d_head)

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-1e20"))
        score = torch.softmax(score, dim=-1)
        out = score @ v

        ## concat
        out = self.concat(out)

        ## linear
        out = self.w_concat(out)

        return out

    def split(self, in_tensor):
        ## input shape : [N, L, embed_size]
        ## output shape : [N, num_heads, L, head_dim]

        N, L, _ = in_tensor.size()
        out_tensor = in_tensor.view(N, L, self.nheads, self.d_head).transpose(1, 2)


        return out_tensor

    def concat(self, in_tensor):
        ## inverse of split method
        ## input shape : [N, num_heads, L, head_dim]
        ## output shape : [N, L, embed_size]
        N = in_tensor.shape[0]
        L = in_tensor.shape[2]

        out_tensor = in_tensor.transpose(1, 2).contiguous().view(N, L, self.d_model)
        return out_tensor

