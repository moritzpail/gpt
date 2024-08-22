import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseLM


class BigramLanguageModel(BaseLM):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(
            self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # idx and targets are both of shape (B, T)
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
        