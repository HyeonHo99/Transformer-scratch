import torch.nn as nn
from blocks import EncoderBlock
from PositionalEmbedding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, nlayers, d_model, nheads, d_feedforward, dropout, device):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(max_len=max_len, d_model=d_model, device=device)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model, nheads=nheads, d_feedforward=d_feedforward, dropout=dropout) for _ in
            range(nlayers)
        ])

    def forward(self, x, mask):
        x = self.dropout(self.embedding(x) + self.pos_embedding(x))

        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x

