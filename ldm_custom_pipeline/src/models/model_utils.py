# Standard libraries
import os

# Third-party libraries
import torch

# Local application/modules
from src.constants.types import Optional, Tuple


def save_checkpoint(
        epoch:          int, 
        model:          torch.nn.Module, 
        optimizer:      torch.optim.Optimizer, 
        scheduler:      Optional[torch.optim.lr_scheduler._LRScheduler], 
        loss:           float, 
        filename:       str
    ) ->                None:
    """
    Save model, optimizer, scheduler, and loss states as a checkpoint.

    :param epoch: Current epoch.
    :param model: Model instance to be saved.
    :param optimizer: Optimizer instance to be saved.
    :param scheduler: Learning rate scheduler instance to be saved.
    :param loss: Current loss value.
    :param filename: Path to save the checkpoint.
    """

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(checkpoint, filename)



def load_checkpoint(
        model:      torch.nn.Module, 
        optimizer:  torch.optim.Optimizer, 
        scheduler:  Optional[torch.optim.lr_scheduler._LRScheduler], 
        filename:   str
    ) ->            Tuple[Optional[int], torch.nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler], Optional[float]]:
    """
    Load model, optimizer, scheduler, and loss states from a checkpoint.

    :param model: Model instance to be loaded.
    :param optimizer: Optimizer instance to be loaded.
    :param scheduler: Learning rate scheduler instance to be loaded.
    :param filename: Path of the checkpoint.

    :return: Tuple containing loaded epoch, model, optimizer, scheduler, and loss or None if checkpoint does not exist.
    """

    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch, model, optimizer, scheduler, loss
    else:
        print("No checkpoint found.")
        return None, model, optimizer, scheduler, None
