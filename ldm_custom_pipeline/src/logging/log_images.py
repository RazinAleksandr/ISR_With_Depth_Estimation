# Standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Third-party libraries
import torch
import torchvision.utils as vutils

import mlflow
import wandb

# Local application/modules
from src.constants.types import Dict, Any, List, Optional
from src.utils.helpers import create_folder_if_not_exists


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
    create_folder_if_not_exists(f'{logdir}/test_samples/epoch_{epoch}')
    for idx, (image, grid_dict) in enumerate(zip(decoded_samples, noise_samples)):
        # Save the image to a file
        file_name = f"reconstructed_epoch_{epoch}_sample_{idx}.png"
        file_path = os.path.join(f'{logdir}/test_samples/epoch_{epoch}', file_name)

        grid = vutils.make_grid(image.detach().cpu(), nrow=image.shape[0] // 2, normalize=True, padding=2, pad_value=1)
        vutils.save_image(grid, file_path)

        n = len(grid_dict['samples'])
        cols = n // 2
        rows = np.ceil(n / cols).astype(int)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

        # Flatten axs to easily iterate over it
        axs = axs.ravel()

        for i in range(n):
            axs[i].imshow(grid_dict['samples'][i])
            axs[i].set_title(f"Current x (step {grid_dict['steps'][i]})")
            axs[i].axis('off')  # Optional: to turn off axis ticks and labels

        # If there are any extra axes, we should turn them off.
        for j in range(n, rows * cols):
            axs[j].axis('off')

        plt.tight_layout()

        # Save the concatenated image as a file
        concatenated_file_name = f"unet_epoch_{epoch}_sample_{idx}.png"
        concatenated_file_path = os.path.join(f'{logdir}/test_samples/epoch_{epoch}', concatenated_file_name)
        plt.savefig(concatenated_file_path)

        # Log the concatenated image file to MLflow
        if log == "mlflow":
            mlflow.log_artifact(file_path)
            mlflow.log_artifact(concatenated_file_path)
        elif log == "wandb":
            wandb.log({"reconstructed_images": [wandb.Image(grid, caption=file_name)], 
                       "concatenated_images": [wandb.Image(fig)]})