# Standard libraries
import os

# Third-party libraries
import torch
import torchvision.utils as vutils

# Local application/modules
from src.constants.types import Dict, Any, List, Tensor
from src.utils.helpers import create_folder_if_not_exists


def save_and_log_images(decoded_samples: Tensor, noise_samples: List[Dict[str, Any]], epoch: int, 
                        logdir: str) -> None:
    """
    Save decoded and noise samples to disk and log them to specified platform.

    :param decoded_samples: Tensor containing decoded image samples.
    :param noise_samples: List containing dictionaries with noise sample data.
    :param epoch: Current epoch.
    :param logdir: Directory to save the images.
    """
    create_folder_if_not_exists(f'{logdir}/test_samples')
    final_grid_images, final_grid_noises = [], []
    for image, grid_dict in zip(decoded_samples, noise_samples):
        targets = image[:image.shape[0]//2, :, :, :]
        predictions = image[image.shape[0]//2:, :, :, :]

        # Target / predictions decoded grid
        final_grid_images.append(
            torch.cat(
                [vutils.make_grid(targets, nrow=8),
                vutils.make_grid(predictions, nrow=8)],
                dim=1
                )
                    )
        
        # Get images from noise_samples[i]
        batch_list = []
        for grid in grid_dict['samples']:
            batch_list.append(grid.permute(2, 0, 1))  # Convert HxWxC to CxHxW for torchvision utils
        batch_grid = torch.cat(batch_list, dim=1)
        batch_grid = vutils.make_grid(
            batch_grid.detach().cpu(), 
            nrow=batch_grid.shape[0] // 2, 
            normalize=True, 
            padding=10, 
            pad_value=1 # White padding
            )
        
        final_grid_noises.append(batch_grid)
    
    # Save image predictions
    final_grid_im = torch.cat(
        [torch.cat(final_grid_images[:len(final_grid_images)//2], dim=1),
        torch.cat(final_grid_images[len(final_grid_images)//2:], dim=1)],
        dim=2
    )
    im_file_name = f"im-pred_epoch-{epoch}.png"
    im_file_path = os.path.join(f'{logdir}/test_samples', im_file_name)
    vutils.save_image(final_grid_im, im_file_path)

    # Save noise predictions
    final_grid_n = torch.cat(final_grid_noises, dim=2)
    n_file_name = f"noise-pred_epoch-{epoch}.png"
    n_file_path = os.path.join(f'{logdir}/test_samples', n_file_name)
    vutils.save_image(final_grid_n, n_file_path)

    image_logger_input = {
        "image_predictions": {
            "image": final_grid_im,
            "caption": im_file_name,
            "file_path": im_file_path
        },
        "noise_predictions": {
            "image": final_grid_n,
            "caption": n_file_name,
            "file_path": n_file_path
        }
    }

    return image_logger_input
