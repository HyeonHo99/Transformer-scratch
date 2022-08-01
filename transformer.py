import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, nlayers, d_model, nheads, d_feedforward, dropout, device, max_len):
        super(Transformer,self).__init__()
        self.encoder = Encoder(vocab_size=src_vocab_size, max_len=max_len, nlayers=nlayers, d_model=d_model,
                               nheads=nheads, d_feedforward=d_feedforward, dropout=dropout, device=device)
        self.decoder = Decoder(vocab_size=trg_vocab_size, max_len=max_len, nlayers=nlayers, d_model=d_model,
                               nheads=nheads, d_feedforward=d_feedforward, dropout=dropout, device=device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        ## shape: [N,L] -> [N,1,1,L]
        src_mask = (src != self.src_pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)

        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N,trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len)

        return trg_mask.to(self.device)

    def forward(self, src_in, trg_in):
        src_mask = self.make_src_mask(src_in)
        trg_mask = self.make_trg_mask(trg_in)
        enc_out = self.encoder(src_in,src_mask)
        dec_out = self.decoder(x=trg_in, enc_out=enc_out, src_mask=src_mask, trg_mask=trg_mask)

        return dec_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_in = torch.tensor([[1,2,3,4,5,6,0,0,0,0],[1,2,3,4,5,0,0,0,0,0]]).to(device)
    trg_in = torch.tensor([[1,2,3,4,5,6,7,0,0,0],[1,2,3,4,5,6,7,8,0,0]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size, src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx, nlayers=2, d_model=64, nheads=4, d_feedforward=128,
                        dropout=0.2, device=device, max_len=10).to(device)

    out = model(src_in,trg_in)

    print(out.shape)
    print(out[0].shape)
    print(out[1].shape)