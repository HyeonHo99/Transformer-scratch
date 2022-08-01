import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

# d_model: the number of expected features in the encoder/decoder inputs
# nheads: the number of heads in the multiheadattention models
# d_feedforward: the dimension of the feedforward network model

class EncoderBlock(nn.Module):
    def __init__(self, d_model, nheads, d_feedforward, dropout):
        super(EncoderBlock,self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MHA = MultiHeadAttention(d_model, nheads)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v, mask):
        _x = q
        x = self.dropout(self.MHA(q=q, k=k, v=v, mask=mask))
        _x = self.norm1(x + _x)
        x = self.feed_forward(x)
        out = self.norm2(x + _x)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nheads, d_feedforward, dropout):
        super(DecoderBlock,self).__init__()
        self.MHA = MultiHeadAttention(d_model, nheads)
        self.norm = nn.LayerNorm(d_model)
        self.transformer_block = EncoderBlock(d_model, nheads, d_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, src_mask, trg_mask):
        q = self.dropout(self.norm(self.MHA(q=q, k=q, v=q, mask=trg_mask) + q))
        out = self.transformer_block(q=q, k=k, v=v, mask=src_mask)

        return out