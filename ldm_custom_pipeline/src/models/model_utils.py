# Standard libraries
import os

# Third-party libraries
import torch

# Local application/modules
from src.constants.types import (Optional, Tuple, Any, 
                                 Optimizer, Module, _LRScheduler)


def save_checkpoint(
        epoch:          int, 
        model:          Module, 
        optimizer:      Optimizer, 
        scheduler:      Optional[_LRScheduler], 
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
        model:      Module, 
        optimizer:  Optimizer, 
        scheduler:  Optional[_LRScheduler], 
        filename:   str
    ) ->            Tuple[Optional[int], Module, Optimizer, Optional[_LRScheduler], Optional[float]]:
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
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch, model, optimizer, scheduler, loss
    else:
        print("No checkpoint found.")
        return None, model, optimizer, scheduler, None


def load_model_and_optimizer(unet: Module, 
                             opt: Optimizer, 
                             lr_scheduler: Any, 
                             checkpoint_resume: str) -> Tuple[int, Module, Optimizer, Any]:
    """
    Load the model and optimizer state from a checkpoint if provided.

    :param unet: The U-Net model.
    :param opt: Optimizer for the model.
    :param lr_scheduler: Learning rate scheduler for the optimizer.
    :param checkpoint_resume: Path to the checkpoint to resume from.

    :return: Tuple containing the start epoch, updated U-Net model, updated optimizer, and updated learning rate scheduler.
    """
    
    start_epoch = 0
    if checkpoint_resume:
        start_epoch, unet, opt, scheduler, _ = load_checkpoint(unet, opt, lr_scheduler, checkpoint_resume)
        
        if lr_scheduler:
            lr_scheduler.optimizer = opt
        else:
            lr_scheduler = scheduler

        print(f'Resume train from {start_epoch}, checkpoint_path: {checkpoint_resume}')

    return start_epoch, unet, opt, lr_scheduler


def get_lr(optimizer):
    """
    Get current learning rate value.

    :param optimizer: Current optimizer.
    
    :return: Learning rate value.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
