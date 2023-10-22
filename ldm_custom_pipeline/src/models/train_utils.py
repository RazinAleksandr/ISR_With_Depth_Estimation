# Standard libraries
import os
from tqdm import tqdm

# Third-party libraries
import mlflow
import torch
import wandb

# Local application/modules
from src.constants.types import Any, Optional, Tuple, List
from src.metrics.PSNR import PSNR
from src.models.eval_utils import eval
from src.models.model_utils import save_checkpoint
from src.logging.log_images import save_and_log_images


# Single train step
def train_one_batch(
        unet:                   torch.nn.Module,
        vae:                    torch.nn.Module,
        images:                 torch.Tensor,
        degradations:           torch.Tensor,
        depths:                 torch.Tensor,
        opt:                    torch.optim.Optimizer,
        device:                 torch.device,
        num_train_timesteps:    int,
        noise_scheduler:        Any,
        loss_fn:                torch.nn.Module,
        epoch:                  int,
        log:                    Optional[str] = None
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
    :param epoch: Current training epoch.
    :param log: The logging method to use ("mlflow" or "wandb"). Default is None.

    :return: Loss value for the batch.
    """

    images = images.to(device) * 2 - 1
    degradations = degradations.to(device)
    depths = depths.to(device)
        
    with torch.no_grad():
        latents = 0.18215 * vae.encode(images).latents

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, num_train_timesteps-1, (latents.shape[0],)).long().to(device)

    latents_input = torch.cat((latents, degradations, depths), 1) 
    noisy_latents = noise_scheduler.add_noise(latents_input, noise, timesteps)
    # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
    # pred = unet(noisy_latents, timesteps, depths, degradations)
    pred = unet(noisy_latents, timesteps)
    loss = loss_fn(pred, noise)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if log == "mlflow":
        mlflow.log_metric("train_iteration_loss", loss.item(), step=epoch)
    elif log == "wandb":
        wandb.log({"train_iteration_loss": loss.item(), "epoch": epoch})
    
    return loss.item()


# Validation loop
def validate(
        unet:                   torch.nn.Module,
        vae:                    torch.nn.Module,
        dataloader:             torch.utils.data.DataLoader,
        device:                 torch.device,
        num_train_timesteps:    int,
        noise_scheduler:        Any,
        loss_fn:                torch.nn.Module
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
            degradations = degradations.to(device)
            depths = depths.to(device)
            
            latents = 0.18215 * vae.encode(images).latents
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, num_train_timesteps-1, (latents.shape[0],)).long().to(device)
            
            latents_input = torch.cat((latents, degradations, depths), 1) 
            noisy_latents = noise_scheduler.add_noise(latents_input, noise, timesteps)
            # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # pred = unet(noisy_latents, timesteps, depths, degradations)
            pred = unet(noisy_latents, timesteps)
            loss = loss_fn(pred, noise)
            
            losses.append(loss.item())

    return sum(losses) / len(losses)


# Training and validation loop
def fit(
        n_epochs:                   int,
        device:                     torch.device,
        num_train_timesteps:        int,
        unet:                       torch.nn.Module,
        vae:                        torch.nn.Module,
        train_dataloader:           torch.utils.data.DataLoader,
        val_dataloader:             torch.utils.data.DataLoader,
        test_dataloader:            torch.utils.data.DataLoader,
        opt:                        torch.optim.Optimizer,
        noise_scheduler:            Any,
        loss_fn:                    torch.nn.Module,
        logdir:                     str = None,
        log:                        Optional[str] = None,
        model_name:                 str = None,
        val_step:                   int = 400,
        start_epoch:                int = 0,
        test_metric:                Optional[Any] = PSNR,
        num_inference_steps:        Optional[int] = None
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
    :param logdir: Directory where logs will be saved.
    :param log: Type of logging used. Options include "mlflow", "wandb", or None.
    :param model_name: Name of the model. Used for saving the model checkpoint.
    :param val_step: Number of iterations after which validation is performed.
    :param start_epoch: Epoch number to start training from (useful for resuming training).
    :param test_metric: Metric function used to evaluate the test performance.
    :param num_inference_steps: Number of inference steps during testing.

    :return: Tuple containing lists of training and validation losses over epochs.
    """

    num_inference_steps = num_train_timesteps if not num_inference_steps else num_inference_steps
    train_loss_history, val_loss_history = [], []
    best_loss = float('inf')  # For checkpoint saving

    total_iterations = len(train_dataloader)
    iteration = 0  # Initialize iteration counter
    
    for epoch in range(1 + start_epoch, n_epochs + 1 + start_epoch):
        epoch_train_losses, epoch_val_losses = [], []
        pbar = tqdm(total=total_iterations, desc=f"Epoch {epoch}/{n_epochs}")
        
        for images, (degradations, depths) in train_dataloader:
            # train step
            unet.train()
            train_loss = train_one_batch(
                unet, vae, images, degradations, depths, opt, device,
                num_train_timesteps, noise_scheduler, loss_fn, epoch, log
            )
            
            iteration += 1
            pbar.update(1)

            # validation loop
            if iteration % val_step == 0:
                val_loss = validate(
                    unet, vae, val_dataloader, device, num_train_timesteps,
                    noise_scheduler, loss_fn
                )
                
                if log == "mlflow":
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                elif log == "wandb":
                    wandb.log({"val_loss": val_loss, "epoch": epoch})
                
                epoch_val_losses.append(val_loss)

            epoch_train_losses.append(train_loss)

        # evaluation loop
        test_prediction, mean_test_metric, noise_samples = eval(
            unet, vae, test_dataloader, device, num_inference_steps, 
            noise_scheduler, test_metric, epoch, log
        )
        
        # log images
        save_and_log_images(test_prediction, noise_samples, epoch, logdir, log)
        pbar.close()

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

        # update model checkpoint
        if model_name and (avg_val_loss < best_loss):
            best_loss = avg_val_loss
            save_checkpoint(
                epoch, unet, opt, avg_val_loss,
                os.path.join(f'{logdir}/models', f'{model_name}-{epoch}_epoch.pth')
            )

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch}/{n_epochs} => "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Test metric {mean_test_metric:.4f}\n")
    
    return train_loss_history, val_loss_history




    
