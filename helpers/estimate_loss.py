import torch
import torch.nn as nn

from helpers.get_batch import get_batch


@torch.no_grad()
def estimate_loss(
    model: nn.Module, 
    train_data: torch.Tensor, 
    valid_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the loss of a given model on the training and validation datasets.

    Args:
        model (nn.Module): The model to estimate the loss for.
        train_data (torch.Tensor): The training dataset.
        valid_data (torch.Tensor): The validation dataset.
        block_size (int): The size of the block to use for the dataset.
        batch_size (int): The size of the batch to use for the dataset.
        eval_iters (int): The number of iterations to evaluate the model.
        device (torch.device): The device to use for the model.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the average loss on the training and validation datasets.
    """
    model.eval()
    out = {}
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = train_data if split == 'train' else valid_data
            X, Y = get_batch(data, block_size, batch_size, device=device)
            _logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out['train'], out['valid']