# Small FT-Transformer (Gorishniy et al., 2021) for numeric tabular features.
# Attention is implemented explicitly (Linear Q/K/V, softmax(QK^T / sqrt(d_head)) V) with no fused
# SDPA / flash kernels, so that:
#   (a) the attention weights of every head and layer can be returned, and
#   (b) the whole graph supports double backprop (needed for gradient-based attacks / explanations later).
# Token 0 is the CLS token; tokens 1..F are the F input features in feature order.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    # token_j = x_j * W_j + b_j for each numeric feature j, plus a learned CLS token.
    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        bound = 1 / math.sqrt(d_model)
        for p in (self.weight, self.bias, self.cls):
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class ExplicitMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Softmax in fp32 even under autocast.
        attn = torch.softmax(scores.float(), dim=-1)
        ctx = torch.matmul(self.dropout(attn).to(v.dtype), v)
        ctx = ctx.transpose(1, 2).reshape(B, T, D)
        return self.out(ctx), attn


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class Block(nn.Module):
    # Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x)).
    def __init__(self, d_model, n_heads, ffn_factor, attn_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        d_ffn = int(d_model * ffn_factor)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ExplicitMultiHeadAttention(d_model, n_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_ffn), ReGLU(), nn.Dropout(ffn_dropout),
                                 nn.Linear(d_ffn, d_model))
        self.res_drop = nn.Dropout(residual_dropout)

    def forward(self, x):
        h, attn = self.attn(self.ln1(x))
        x = x + self.res_drop(h)
        x = x + self.res_drop(self.ffn(self.ln2(x)))
        return x, attn


class FTTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=96, n_blocks=3, n_heads=4, ffn_factor=4 / 3,
                 attn_dropout=0.1, ffn_dropout=0.1, residual_dropout=0.0):
        super().__init__()
        self.kwargs = dict(n_features=n_features, n_classes=n_classes, d_model=d_model,
                           n_blocks=n_blocks, n_heads=n_heads, ffn_factor=ffn_factor,
                           attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
                           residual_dropout=residual_dropout)
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffn_factor, attn_dropout, ffn_dropout,
                                           residual_dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, n_classes))

    def forward(self, x, return_attn=False):
        # x: (B, F) preprocessed features. Returns logits (B, C) and, if requested,
        # a list with one (B, n_heads, F+1, F+1) attention tensor per block (not detached).
        h = self.tokenizer(x)
        attns = []
        for block in self.blocks:
            h, a = block(h)
            if return_attn:
                attns.append(a)
        logits = self.head(h[:, 0])
        return (logits, attns) if return_attn else logits
