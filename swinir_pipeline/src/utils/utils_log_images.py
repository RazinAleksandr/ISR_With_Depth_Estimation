# Standard libraries
import os

# Third-party libraries
import torch
import torchvision.utils as vutils


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


def save_and_log_images(decoded_samples, epoch: int, logdir: str) -> None:
    """
    Save decoded and noise samples to disk and log them to specified platform.

    :param decoded_samples: Tensor containing decoded image samples.
    :param epoch: Current epoch.
    :param logdir: Directory to save the images.
    """
    create_folder_if_not_exists(f'{logdir}/test_samples')
    final_grid_images = []
    for image in decoded_samples:
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
    
    # Save image predictions
    final_grid_im = torch.cat(
        [torch.cat(final_grid_images[:len(final_grid_images)//2], dim=1),
        torch.cat(final_grid_images[len(final_grid_images)//2:], dim=1)],
        dim=2
    )
    im_file_name = f"im-pred_epoch-{epoch}.png"
    im_file_path = os.path.join(f'{logdir}/test_samples', im_file_name)
    vutils.save_image(final_grid_im, im_file_path)

   
    image_logger_input = {
            "image": final_grid_im,
            "caption": im_file_name,
            "file_path": im_file_path
        }

    return image_logger_input
