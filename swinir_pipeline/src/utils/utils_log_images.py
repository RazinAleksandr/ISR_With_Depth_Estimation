# Standard libraries
import os
from PIL import Image
import numpy as np


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


def save_and_log_images(predictions, targets, epoch: int, logdir: str) -> None:
    """
    Save decoded and noise samples to disk and log them to specified platform.

    :param decoded_samples: Tensor containing decoded image samples.
    :param epoch: Current epoch.
    :param logdir: Directory to save the images.
    """
    create_folder_if_not_exists(f'{logdir}/test_samples')

    grid_shape=(len(predictions) // 10, 10)
    
    # Create rows of targets and predictions
    target_rows = []
    prediction_rows = []

    for i in range(0, len(targets), grid_shape[1]):
      target_rows.append(np.concatenate(targets[i:i + grid_shape[1]], axis=1))
      prediction_rows.append(np.concatenate(predictions[i:i + grid_shape[1]], axis=1))

      # Concatenate all rows to form the grid
      grid_rows = []
      for target_row, prediction_row in zip(target_rows, prediction_rows):
        grid_rows.append(target_row)
        grid_rows.append(prediction_row)
    final = np.concatenate(grid_rows, axis=0)
    final_image = Image.fromarray(final.astype('uint8'))

    # Save image
    im_file_name = f"im-pred_epoch-{epoch}.png"
    im_file_path = os.path.join(f'{logdir}/test_samples', im_file_name)
    final_image.save(im_file_path)

    image_logger_input = {
        "image": final_image,
        "caption": im_file_name,
        "file_path": im_file_path
    }

    return image_logger_input
