import torch 


def get_batch(
        data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a batch of random sequences from the given data.

    Parameters:
    data (torch.Tensor): The input data to sample from.
    block_size (int): The length of each sequence in the batch.
    batch_size (int): The number of sequences in the batch.
    device (torch.device): The device to move the batch to.

    Returns:
    tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors, x and y, where x is the batch of sequences and y is the batch of sequences shifted by one position.
    """
    n = len(data)
    ix = torch.randint(n - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y