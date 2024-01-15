import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # heads: number of self-attention heads
        # d_embed: dimension of input features

        # represents Wq,Wk,Wv matrices
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.heads = heads
        self.d_head = d_embed // heads

    def forward(self, x, causal_mask=False):
        # x: (seq, dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.heads, self.d_head)

        # (seq, dim) -> (seq, 3*dim) -> 3*(seq, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (seq, dim) -> (seq, h, dim / h) -> (h, seq, dim / h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (h, seq, seq)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask upper triangle with -inf, so dies with softmax
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (h, seq, dim / h)
        output = weight @ v

        # (seq, h, dim / h)
        output = output.transpose(1, 2)

        # (seq, dim)
        output = output.reshape(input_shape)

        # (seq, dim)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # heads: number of attention heads
        # d_embed: dimension of input feature embeddings
        # d_cross: dimension of context feature embeddings

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.heads = heads
        self.d_heads = d_embed // heads

    def forward(self, x, y):
        # x: (seq, dim) latent reps queries
        # y: (seq, dim) context reps key/values
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.heads, self.d_heads)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (seq, dim) -> (seq, h, dim / h) -> (h, seq, dim / h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (h, seq, seq)
        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=-1)

        # (h, seq, dim / h)
        output = weight @ v

        # (seq, h, dim / h)
        output = output.transpose(1, 2).contiguous()

        # (seq, dim)
        output = output.view(input_shape)

        # (seq, dim)
        output = self.out_proj(output)

        return output
