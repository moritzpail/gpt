import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseLM


class GPT(BaseLM):
    def __init__(
        self, 
        vocab_size: int, 
        n_embed_size: int, 
        block_size: int, 
        device: torch.device,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.n_embed_size = n_embed_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embed_size)
        self.blocks = nn.Sequential(
            *[GPTBlock(n_heads, n_embed_size, block_size, dropout_rate) for _ in range(n_layers)],
        )
        self.ln_f = nn.LayerNorm(n_embed_size)
        self.lm_head = nn.Linear(n_embed_size, vocab_size)
        self.device = device
    
    def forward(
            self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape

        # idx and targets are both of shape (B, T)
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embed_size)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )
        x = tok_emb + pos_emb # (B, T, n_embed_size)
        x = self.blocks(x) # (B, T, n_embed_size)
        x = self.ln_f(x) # (B, T, n_embed_size)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class GPTBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed_size: int, block_size: int, dropout_rate: float = 0.1):
        super().__init__()
        head_size = n_embed_size // n_heads
        self.attn = MultiHeadAttention(n_heads, head_size, n_embed_size, block_size, dropout_rate)
        self.mlp = FeedForward(n_embed_size, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(n_embed_size)
        self.layer_norm2 = nn.LayerNorm(n_embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embed_size: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed_size,4 * n_embed_size),
            nn.ReLU(),
            nn.Linear(4 * n_embed_size, n_embed_size), # Projection layer
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            head_size: int, 
            n_embed_size: int, 
            block_size: int,
            dropout_rate: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size, n_embed_size, block_size, dropout_rate) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed_size, n_embed_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size: int, n_embed_size: int, block_size: int, dropout_rate: float):
        super().__init__()
        self.key = nn.Linear(n_embed_size, head_size, bias=False)
        self.query = nn.Linear(n_embed_size, head_size, bias=False)
        self.value = nn.Linear(n_embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
