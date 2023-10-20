# Standard libraries
import os

import matplotlib.pyplot as plt

# Third-party libraries
import torch
import torchvision.utils as vutils

import mlflow
import wandb

# Local application/modules
from src.constants.types import Dict, Any, List, Optional


def save_and_log_images(decoded_samples: torch.Tensor, noise_samples: List[Dict[str, Any]], epoch: int, 
                        logdir: str, log: Optional[str] = False) -> None:
    """
    Save decoded and noise samples to disk and log them to specified platform.

    :param decoded_samples: Tensor containing decoded image samples.
    :param noise_samples: List containing dictionaries with noise sample data.
    :param epoch: Current epoch.
    :param logdir: Directory to save the images.
    :param log: Specifies which platform to log to, options are "mlflow", "wandb" or False for no logging.
    """

    for idx, (image, grid_dict) in enumerate(zip(decoded_samples, noise_samples)):
        # Save the image to a file
        file_name = f"reconstructed_epoch_{epoch}_sample_{idx}.png"
        file_path = os.path.join(f'{logdir}/test_samples', file_name)
        vutils.save_image(image, file_path)

        n = len(grid_dict['samples'])
        fig, axs = plt.subplots(n, 1, figsize=(12, 5))
        for i in range(len(grid_dict['samples'])):
            axs[i].imshow(grid_dict['samples'][i])
            axs[i].set_title(f"Current x (step {grid_dict['steps'][i]})")
        plt.tight_layout()

        # Save the concatenated image as a file
        concatenated_file_name = f"unet_epoch_{epoch}_sample_{idx}.png"
        concatenated_file_path = os.path.join(logdir, concatenated_file_name)
        plt.savefig(concatenated_file_path)

        # Log the concatenated image file to MLflow
        if log == "mlflow":
            mlflow.log_artifact(file_path)
            mlflow.log_artifact(concatenated_file_path)
        elif log == "wandb":
            wandb.log({"reconstructed_images": [wandb.Image(image, caption=file_name)], 
                       "concatenated_images": [wandb.Image(fig)]})