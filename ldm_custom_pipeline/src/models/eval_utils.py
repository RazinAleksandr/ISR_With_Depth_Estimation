# Standard libraries
from tqdm import tqdm
import inspect

# Third-party libraries
import mlflow
import torch
import torchvision
import wandb

# Local application/modules
from src.constants.types import Any, Optional, Tuple, List


# Test evaluation loop
def eval(    
        unet:                   torch.nn.Module,
        vae:                    torch.nn.Module,
        dataloader:             torch.utils.data.DataLoader,
        device:                 torch.device,
        num_inference_steps:    int,
        noise_scheduler:        Any,
        test_metric:            Optional[Any],
        epoch:                  int,
        log:                    Optional[str] = None
    ) ->                        Tuple[List[torch.Tensor], float, List[Any]]:
    """
    Test the model using a provided dataloader and compute the specified test metric.

    :param unet: The U-Net model used for predictions.
    :param vae: The Variational AutoEncoder model used for encoding and decoding.
    :param dataloader: DataLoader providing test data batches.
    :param device: Computational device (CPU or GPU) the model runs on.
    :param num_inference_steps: Number of inference steps during testing.
    :param noise_scheduler: Scheduler used to manage and apply noise during testing.
    :param test_metric: Metric function used to evaluate the test performance. 
    :param epoch: Current epoch number.
    :param log: Type of logging used. Options include "mlflow", "wandb", or None.

    :return: Decoded samples, mean test metric value, and a list containing noise samples.
    """

    unet.eval()  # Set the model to evaluation mode
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    decoded_samples, noise_samples = [], []
    test_metric_values = []  # List to store PSNR values for each image
    with torch.no_grad():
        for images, (degradations, depths) in tqdm(dataloader, desc='Test'):
            images = images.to(device) # * 2 - 1 # mapped to (-1, 1) # we sample from normal distr, so it is not necessary
            degradations = degradations.to(device)
            depths = depths.to(device)
            
            # The random starting point
            latents = torch.randn_like(degradations).to(device)
            latents = latents * noise_scheduler.init_noise_sigma

            eta = 1
            accepts_eta = "eta" in set(inspect.signature(noise_scheduler.step).parameters.keys())
            extra_kwargs = {}
            if accepts_eta:
                extra_kwargs["eta"] = eta

            # Initialize an empty lists to store individual grids
            grid_dict = {'samples': [], 'steps': []}
            # Loop through the sampling timesteps
            for i, t in enumerate(noise_scheduler.timesteps):
                latents_input = torch.cat((latents, degradations, depths), 1) 
                latents_input = noise_scheduler.scale_model_input(latents_input, t)
                
                # Get the prediction
                noise_pred = unet(latents_input, t)

                # Calculate what the updated sample should look like with the scheduler
                scheduler_output = noise_scheduler.step(noise_pred, t, latents,  **extra_kwargs)

                # Update latents
                latents = scheduler_output.prev_sample

                # Occasionally add the grid to the list
                if i % ( int(len(noise_scheduler.timesteps)*0.1) ) == 0 or i == len(noise_scheduler.timesteps) - 1:
                    grid = torchvision.utils.make_grid(latents, nrow=4).permute(1, 2, 0)
                    grid_dict['samples'].append(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
                    grid_dict['steps'].append(i)

            decoded = vae.decode(latents).sample
            decoded = torch.clamp(decoded, -1.0, 1.0) # return image to adequat dist
            decoded = decoded / 2 + 0.5
            # Concatenate them along the batch dimension
            decoded_sample = torch.cat((images, decoded), 0)

            # Compute PSNR for each image in the batch and store it
            for orig, recon in zip(images, decoded):
                test_metric_value = test_metric(orig, recon)
                test_metric_values.append(test_metric_value)

            noise_samples.append(grid_dict)
            decoded_samples.append(decoded_sample)
    
    # Calculate mean PSNR for all test batches
    mean_test_metric = sum(test_metric_values) / len(test_metric_values)
    
    if log == "mlflow":
        mlflow.log_metric("mean_test_metric", mean_test_metric, step=epoch)
    elif log == "wandb":
        wandb.log({"mean_test_metric": mean_test_metric, "epoch": epoch})

    return decoded_samples, mean_test_metric, noise_samples