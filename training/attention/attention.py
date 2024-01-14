import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # represents Wq,Wk,Wv matrices
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.heads = heads
        self.d_head = d_embed // heads

    def forward(self, x, causal_mask=False):
        # x: (batch, seq, dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.heads, self.d_head)

        # (batch, seq, dim) -> (batch, seq, 3*dim) -> 3*(batch, seq, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch, seq, dim) -> (batch, seq, h, dim / h) -> (batch, h, seq, dim / h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch, h, seq, seq)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask upper triangle with -inf, so dies with softmax
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch, h, seq, dim / h)
        output = weight @ v

        # (batch, seq, h, dim / h)
        output = output.transpose(1, 2)

        # (batch, seq, dim)
        output = output.reshape(input_shape)

        # (batch, seq, dim)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(
        self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x, y):
        # x: (batch, seq, dim) latent reps queries
        # y: (batch, seq, dim) context reps key/values
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.q_proj(y)
        v = self.q_proj(y)

        # (batch, seq, dim) -> (batch, seq, h, dim / h) -> (batch, h, seq, dim / h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch, h, seq, seq)
        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch, h, seq, dim / h)
        output = weight @ v

        # (batch, seq, h, dim / h)
        output = output.transpose(1, 2).contiguous()

        # (batch, seq, dim)
        output = output.view(input_shape)

        # (batch, seq, dim)
        output = self.out_proj(output)

        return output
