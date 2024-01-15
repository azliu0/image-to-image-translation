import sys

sys.path.append(".")
sys.path.append("..")

import torch
from torch import nn
from torch.nn import functional as F
from attention.attention import SelfAttention
from config import MAX_SEQ_LENGTH, TOKEN_EMBEDDING_SIZE, VOCAB_SIZE


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        # n_vocab: size of vocab in tokens
        # n_embd: size of each token embedding
        # n_token: number of tokens to parse in the sequence. default max is given by MAX_SEQ_LENGTH

        # maps n_vocab token keys -> n_embd size embedding vector
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # maps n_token positions -> n_embd size positional embedding vector
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        # tokens: (seq)

        # (seq, dim)
        x = self.token_embedding(tokens)
        # (seq, dim)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        # Norm1
        self.layernorm_1 = nn.LayerNorm(n_embd)

        # Self Attention Head!
        self.attention = SelfAttention(n_head, n_embd)

        # Norm2
        self.layernorm_2 = nn.LayerNorm(n_embd)

        # FFNN
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # x: (seq, dim)

        # first residual block (attention)
        residue = x

        # (seq, dim)
        x = self.layernorm_1(x)

        # (seq, dim)
        x = self.attention(x, causal_mask=True)

        # (seq, dim)
        x += residue

        # second residual block (ffnn)
        residue = x

        # (seq, dim)
        x = self.layernorm_2(x)

        # (seq, 4*dim)
        x = self.linear_1(x)

        # (seq, 4*dim)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        # (seq, dim)
        x = self.linear_2(x)

        # (seq, dim)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(VOCAB_SIZE, TOKEN_EMBEDDING_SIZE, MAX_SEQ_LENGTH)

        self.layers = nn.ModuleList(
            [CLIPLayer(12, TOKEN_EMBEDDING_SIZE) for i in range(12)]
        )

        self.layernorm = nn.LayerNorm(TOKEN_EMBEDDING_SIZE)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens: (seq)
        tokens = tokens.type(torch.long)

        # (seq, dim)
        state = self.embedding(tokens)

        # (seq, dim) for all layers
        for layer in self.layers:
            state = layer(state)

        # (seq, dim)
        output = self.layernorm(state)

        # (seq, dim)
        return output
