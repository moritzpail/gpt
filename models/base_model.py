from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLM(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, idx: torch.Tensor, targets: torch.Tensor=None):
        pass

    def generate(self, idx: int, max_len: int, block_size: int | None = None) -> torch.Tensor:
    # idx is of shape (B, T)
        for _ in range(max_len):

            if block_size is None:
                idx_cond = idx
            else:
                idx_cond = idx[:, -block_size:]
                
            logits, _loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, next_idx], dim=1) # (B, T+1)
        return idx