import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class transformer_base(nn.Module):
    def __init__(self, vocab_size, dim=128, nhead=16, load_pretrain=''):
        super().__init__()
        self.vocab_size = vocab_size
        if load_pretrain:
            print('load pretrained embedding in:{} \n'.format(load_pretrain))
            self.embedding = nn.Embedding.from_pretrained(torch.load(load_pretrain))
            self.embedding.requires_grad_ = True
        else:
            self.embedding = nn.Embedding(self.vocab_size, dim)
        self.PE = PositionalEncoding(dim)
        self.transformer = nn.Transformer(d_model=dim, nhead=nhead, num_encoder_layers=12)
        self.lr_2_vocab = nn.Linear(dim, self.vocab_size)

    def forward(self, src_ids, tgt_ids, src_pad_mask=None, tgt_pad_mask=None, tgt_mask=None):
        src = self.embedding(src_ids)
        tgt = self.embedding(tgt_ids)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        src = self.PE(src)
        tgt = self.PE(tgt)
        out = self.transformer(src=src, tgt=tgt, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask, \
                               memory_key_padding_mask=src_pad_mask, tgt_mask=tgt_mask)
        out = out.permute(1, 0, 2)
        out = self.lr_2_vocab(out)
        return out