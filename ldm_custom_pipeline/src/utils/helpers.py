# Standard libraries
import os
import random

import numpy as np

# Third-party libraries
import torch

# Local application/modules
from src.constants.types import Dict, Any


def str_to_class(exp_config: Dict[str, Any], CLASS_MAPPING: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert string class names in the experiment configuration to actual class references using the provided mapping.

    :param exp_config: Dictionary containing experiment configuration with string class names.
    :param CLASS_MAPPING: Dictionary containing the mapping from string class names to actual class references.

    :return: Updated experiment configuration with actual class references.
    """

    for dataset in ['train_datasets', 'val_datasets', 'test_datasets']:    
        exp_config["datasets"][dataset]['image_dataset'] = CLASS_MAPPING[exp_config["datasets"][dataset]['image_dataset']]
        exp_config["datasets"][dataset]['cond_dataset'] = CLASS_MAPPING[exp_config["datasets"][dataset]['cond_dataset']]
        exp_config["datasets"][dataset]['transform'] = CLASS_MAPPING[exp_config["datasets"][dataset]['transform']]
        exp_config["datasets"][dataset]['degradation'] = CLASS_MAPPING[exp_config["datasets"][dataset]['degradation']]

    exp_config['model']['lr_scheduler'] = CLASS_MAPPING[exp_config['model']['lr_scheduler']]        
    exp_config['model']['pretrain_pipeline'] = CLASS_MAPPING[exp_config['model']['pretrain_pipeline']]
    exp_config['model']['noise_scheduler'] = CLASS_MAPPING[exp_config['model']['noise_scheduler']]
    exp_config['model']['loss'] = CLASS_MAPPING[exp_config['model']['loss']]
    exp_config['model']['test_metric'] = CLASS_MAPPING[exp_config['model']['test_metric']]

    return exp_config


def set_seed(seed: int) -> None:
    """
    Set seeds for randomness sources to ensure reproducibility.

    :param seed: Seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_folder_if_not_exists(folder_path):
    """
    Checks if a folder exists at the specified path and creates it if it doesn't exist.

    Args:
        folder_path (str): The path of the folder to check or create.

    Returns:
        None
    """
    
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{folder_path}': {str(e)}")
    else:
        print(f"Folder '{folder_path}' already exists.")


def epoch_print_info(epoch: int,
                     n_epochs: int,
                     lr: float,
                     avg_train_loss: float,
                     avg_val_loss: float,
                     mean_test_metric: float) -> None:
    """
    Print information related to the training process for the current epoch.

    :param epoch: Current training epoch.
    :param n_epochs: Total number of training epochs.
    :param lg: Learning rate.
    :param avg_train_loss: Average training loss for the current epoch.
    :param avg_val_loss: Average validation loss for the current epoch.
    :param mean_test_metric: Average test metric for the current epoch.

    :return: None
    """
    print(f"Epoch {epoch}/{n_epochs} => "
          f"Learning rate: {lr:.4f}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, "
          f"Test metric {mean_test_metric:.4f}\n")
