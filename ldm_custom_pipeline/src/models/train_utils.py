# Standard libraries
import os
from tqdm import tqdm

# Third-party libraries
import torch

# Local application/modules
from src.constants.types import (Any, Optional, Tuple, List, Module,
                                 Tensor, device, Optimizer, DataLoader, _LRScheduler)
from src.constants.constants import VAE_COEF
from src.metrics.PSNR import PSNR
from src.models.eval_utils import eval
from src.models.model_utils import save_checkpoint, get_lr
from src.logging.log_images import save_and_log_images
from src.utils.helpers import epoch_print_info


# Single train step
def train_one_batch(
        unet:                   Module,
        vae:                    Module,
        images:                 Tensor,
        degradations:           Tensor,
        depths:                 Tensor,
        opt:                    Optimizer,
        device:                 device,
        num_train_timesteps:    int,
        noise_scheduler:        Any,
        loss_fn:                Module,
    ) ->                        float:
    """
    Train the model on a single batch of data.

    :param unet: The U-Net model.
    :param vae: Variational AutoEncoder model.
    :param images: Batch of training images.
    :param degradations: Degradations associated with the images.
    :param depths: Depths associated with the images.
    :param opt: Optimizer for the model.
    :param device: Device to run the model on.
    :param num_train_timesteps: Number of training time steps.
    :param noise_scheduler: Noise scheduler.
    :param loss_fn: Loss function to compute the training loss.

    :return: Loss value for the batch.
    """

    images = images.to(device) * 2 - 1
    degradations = degradations.to(device) * 2 - 1
    depths = depths.to(device) * 2 - 1
        
    with torch.no_grad():
        latents = VAE_COEF * vae.encode(images).latents

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, num_train_timesteps-1, (latents.shape[0],)).long().to(device)

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    latents_input = torch.cat((noisy_latents, degradations, depths), 1) 
        
    pred = unet(latents_input, timesteps)
    loss = loss_fn(pred, noise)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


# Validation loop
def validate(
        unet:                   Module,
        vae:                    Module,
        dataloader:             DataLoader,
        device:                 device,
        num_train_timesteps:    int,
        noise_scheduler:        Any,
        loss_fn:                Module
    ) ->                        float:
    """
    Validate the model on a set of data.

    :param unet: The U-Net model.
    :param vae: Variational AutoEncoder model.
    :param dataloader: Validation dataloader.
    :param device: Device to run the model on.
    :param num_train_timesteps: Number of training time steps.
    :param noise_scheduler: Noise scheduler.
    :param loss_fn: Loss function to compute the training loss.
    
    :return: Average validation loss.
    """
    
    unet.eval()  # Set the model to evaluation mode

    losses = []
    with torch.no_grad():
        for images, (degradations, depths) in tqdm(dataloader, desc="Validation"):
            images = images.to(device) * 2 - 1
            degradations = degradations.to(device) * 2 - 1
            depths = depths.to(device) * 2 - 1
            
            latents = VAE_COEF * vae.encode(images).latents
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, num_train_timesteps-1, (latents.shape[0],)).long().to(device)
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            latents_input = torch.cat((noisy_latents, degradations, depths), 1) 

            pred = unet(latents_input, timesteps)
            loss = loss_fn(pred, noise)
            
            losses.append(loss.item())

    return sum(losses) / len(losses)


# Training and validation loop
def fit(
        n_epochs:                   int,
        device:                     device,
        num_train_timesteps:        int,
        unet:                       Module,
        vae:                        Module,
        train_dataloader:           DataLoader,
        val_dataloader:             DataLoader,
        test_dataloader:            DataLoader,
        opt:                        Optimizer,
        noise_scheduler:            Any,
        loss_fn:                    Module,
        logdir:                     str = None,
        logger:                     Optional[Any] = None,
        model_name:                 str = None,
        val_step:                   int = 400,
        start_epoch:                int = 0,
        test_metric:                Optional[Any] = PSNR,
        num_inference_steps:        Optional[int] = None,
        lr_scheduler:               _LRScheduler = None,
        lr_scheduler_start_epoch:   int = 0,
    ) ->                            Tuple[List[float], List[float]]:
    """
    Train, validate, and test the model for a specified number of epochs.

    :param n_epochs: Number of epochs for training.
    :param device: Computational device (CPU or GPU) the model runs on.
    :param num_train_timesteps: Number of timesteps during training.
    :param unet: The U-Net model used for predictions.
    :param vae: The Variational AutoEncoder model used for encoding and decoding.
    :param train_dataloader: DataLoader providing training data batches.
    :param val_dataloader: DataLoader providing validation data batches.
    :param test_dataloader: DataLoader providing test data batches.
    :param opt: Optimizer for model parameter updates.
    :param noise_scheduler: Scheduler used to manage and apply noise during training.
    :param loss_fn: Loss function used to compute the training loss.
    :param logger: Logger for logging artifacts. Default is None
    :param log: Type of logging used. Options include "mlflow", "wandb", or None.
    :param model_name: Name of the model. Used for saving the model checkpoint.
    :param val_step: Number of iterations after which validation is performed.
    :param start_epoch: Epoch number to start training from (useful for resuming training).
    :param test_metric: Metric function used to evaluate the test performance.
    :param num_inference_steps: Number of inference steps during testing.
    :param lr_scheduler: Learning rate scheduler (epoch steps).
    :param lr_scheduler_start_epoch: From each epoch make shed steps

    :return: Tuple containing lists of training and validation losses over epochs.
    """

    num_inference_steps = num_train_timesteps if not num_inference_steps else num_inference_steps
    train_loss_history, val_loss_history = [], []
    best_loss = float('inf')  # For checkpoint saving

    total_iterations = len(train_dataloader)
    iteration = total_iterations * start_epoch   # Initialize iteration counter
    
    for epoch in range(1 + start_epoch, n_epochs + 1):
        epoch_train_losses, epoch_val_losses = [], []
        pbar = tqdm(total=total_iterations, desc=f"Epoch {epoch}/{n_epochs}")
        
        for images, (degradations, depths) in train_dataloader:
            # Make train step
            unet.train()
            train_loss = train_one_batch(
                unet, vae, images, degradations, depths, opt, device,
                num_train_timesteps, noise_scheduler, loss_fn
            )

            # Log train loss
            if logger:
                logger.log_metric("train_iteration_loss", train_loss, step=iteration, step_name='iter')
            
            iteration += 1
            pbar.update(1)

            # Start validation loop
            if iteration % val_step == 0:
                val_loss = validate(
                    unet, vae, val_dataloader, device, num_train_timesteps,
                    noise_scheduler, loss_fn
                )
                
                # Log val loss
                if logger:
                    logger.log_metric("val_loss", val_loss, step=iteration, step_name='iter')
                
                epoch_val_losses.append(val_loss)

            epoch_train_losses.append(train_loss)

        # evaluation loop
        test_prediction, mean_test_metric, noise_samples = eval(
            unet, vae, test_dataloader, device, num_inference_steps, 
            noise_scheduler, test_metric
        )
        
        # log images
        image_logger_input = save_and_log_images(test_prediction, noise_samples, epoch, logdir)

        # Log learning rate value, eval results
        if logger:
            logger.log_metric("learning_rate", get_lr(opt), step=epoch)
            logger.log_metric("mean_test_metric", mean_test_metric, step=epoch)
            for image_key, v in image_logger_input.items():
                logger.log_image(image_key, **v)

        # Make learning rate step
        if lr_scheduler and epoch >= lr_scheduler_start_epoch:
            lr_scheduler.step()

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

        # update model checkpoint
        if model_name:
            best_loss = min(best_loss, avg_val_loss)
            save_checkpoint(
                epoch, unet, opt, lr_scheduler, avg_val_loss,
                os.path.join(f'{logdir}/models', f'{model_name}-{epoch}_epoch.pth')
            )
        # if model_name and (avg_val_loss < best_loss):
        #     best_loss = avg_val_loss
        #     save_checkpoint(
        #         epoch, unet, opt, lr_scheduler, avg_val_loss,
        #         os.path.join(f'{logdir}/models', f'{model_name}-{epoch}_epoch.pth')
        #     )

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Print info about epoch metrics/losses/lr
        epoch_print_info(epoch, n_epochs, get_lr(opt), avg_train_loss, avg_val_loss, mean_test_metric)

        pbar.close()
    
    return train_loss_history, val_loss_history
