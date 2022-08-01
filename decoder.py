import torch
import torch.nn as nn
from PositionalEmbedding import PositionalEmbedding
from blocks import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, nlayers, d_model, nheads, d_feedforward, dropout, device):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(max_len=max_len, d_model=d_model, device=device)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model=d_model, nheads=nheads, d_feedforward=d_feedforward, dropout=dropout) for _ in range(nlayers)
            ]
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        q = self.dropout(self.embedding(x) + self.pos_embedding(x))

        for layer in self.layers:
            q = layer(q, enc_out, enc_out, src_mask, trg_mask)

        out = self.linear(q)

        return out