import mlflow
import wandb
import torch
from tqdm import tqdm
import os
import torchvision

from src.utils.helpers import save_and_log_images, save_checkpoint
from src.metrics.PSNR import PSNR


def train_one_batch(unet, vae, images, degradations, depths, opt, device, num_train_timesteps, noise_scheduler, loss_fn, epoch, log=None):
    images = images.to(device) * 2 - 1
    degradations = degradations.to(device)
    depths = depths.to(device)
        
    with torch.no_grad():
        latents = 0.18215 * vae.encode(images).latents

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, num_train_timesteps-1, (latents.shape[0],)).long().to(device)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
    pred = unet(noisy_latents, timesteps, depths, degradations)
    loss = loss_fn(pred, noise)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if log == "mlflow":
        mlflow.log_metric("train_iteration_loss", loss.item(), step=epoch)
    elif log == "wandb":
        wandb.log({"train_iteration_loss": loss.item(), "epoch": epoch})
    
    return loss.item()

def validate(unet, vae, dataloader, device, num_train_timesteps, noise_scheduler, loss_fn, epoch, log=None):
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
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            pred = unet(noisy_latents, timesteps, depths, degradations)
            loss = loss_fn(pred, noise)
            
            losses.append(loss.item())

    return sum(losses) / len(losses)

    
# Training and validation loop
def fit(n_epochs, 
        device, 
        num_train_timesteps, 
        unet, 
        vae, 
        train_dataloader,
        val_dataloader,
        test_dataloader,
        opt,
        noise_scheduler,
        loss_fn,
        logdir,
        log=None,
        checkpoint_path=None,
        val_step=400,
        start_epoch=0,
        test_metric=PSNR,
        num_inference_steps=None):

    num_inference_steps =  num_train_timesteps if not num_inference_steps else num_inference_steps
    train_loss_history, val_loss_history = [], []

    best_loss = float('inf') # fpr checkpoint saving

    total_iterations = len(train_dataloader)
    iteration = 0  # Initialize iteration counter
    for epoch in range(1+start_epoch, n_epochs+1+start_epoch):

        epoch_train_losses, epoch_val_losses = [], []
        pbar = tqdm(total=total_iterations, desc=f"Epoch {epoch}/{n_epochs}")
        for images, (degradations, depths) in train_dataloader:
            # Training
            unet.train()
            train_loss = train_one_batch(unet, vae, images, degradations, depths, opt, device, num_train_timesteps, noise_scheduler, loss_fn, epoch, log)
            
            iteration += 1  # Increment iteration counter
            pbar.update(1)

            # Check if it's time for validation
            if iteration % val_step == 0:
                val_loss = validate(unet, vae, val_dataloader, device, num_train_timesteps, noise_scheduler, loss_fn, epoch, log)
                
                if log == "mlflow":
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                elif log == "wandb":
                    wandb.log({"val_loss": val_loss, "epoch": epoch})
                
                epoch_val_losses.append(val_loss)

            epoch_train_losses.append(train_loss)

        # Test and save images after each epoch
        test_prediction, mean_test_metric, noise_samples = test(unet, vae, test_dataloader, device, num_inference_steps, noise_scheduler, test_metric, epoch, log)
        save_and_log_images(test_prediction, noise_samples, epoch, logdir, log)

        pbar.close()

        epoch_train_losses = sum(epoch_train_losses) / len(epoch_train_losses)
        epoch_val_losses = sum(epoch_val_losses) / len(epoch_val_losses)

        # save the best checkpoint
        if checkpoint_path and (epoch_val_losses < best_loss):
            best_loss = epoch_val_losses
            save_checkpoint(epoch, unet, opt, epoch_val_losses, os.path.join(checkpoint_path, f'unet-{epoch}_epoch.pth'))


        train_loss_history.append(epoch_train_losses)
        val_loss_history.append(epoch_val_losses)

        print(f"Epoch {epoch}/{n_epochs} => Train Loss: {epoch_train_losses:.4f}, Validation Loss: {epoch_val_losses:.4f}, Test metric {mean_test_metric:.4f}")
        print(' ')
    
    return train_loss_history, val_loss_history


# def test(unet, vae, dataloader, device, num_train_timesteps, noise_scheduler, test_metric, epoch, log=None):
#     unet.eval()  # Set the model to evaluation mode
    
#     decoded_samples = []
#     test_metric_values = []  # List to store PSNR values for each image
#     with torch.no_grad():
#         for images, (degradations, depths) in tqdm(dataloader, desc='Test'):
#             images = images.to(device) * 2 - 1 # mapped to (-1, 1)
#             degradations = degradations.to(device)
#             depths = depths.to(device)
            
#             latents = 0.18215 * vae.encode(images).latents
            
#             noise = torch.randn_like(latents)
#             timesteps = torch.randint(0, num_train_timesteps-1, (latents.shape[0],)).long().to(device)
#             noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
#             pred = unet(noisy_latents, timesteps, depths, degradations)

#             decoded = vae.decode(pred / 0.18215).sample

#             # Compute PSNR for each image in the batch and store it
#             for orig, recon in zip(images, decoded):
#                 test_metric_value = test_metric(orig, recon)
#                 test_metric_values.append(test_metric_value)

#             decoded_samples.append(decoded)
    
#     # Calculate mean PSNR for all test batches
#     mean_test_metric = sum(test_metric_values) / len(test_metric_values)
    
#     if log == "mlflow":
#         mlflow.log_metric("mean_test_metric", mean_test_metric, step=epoch)
#     elif log == "wandb":
#         wandb.log({"mean_test_metric": mean_test_metric, "epoch": epoch})

#     return decoded_samples, mean_test_metric



def test(unet, vae, dataloader, device, num_inference_steps, noise_scheduler, test_metric, epoch, log=None):
    unet.eval()  # Set the model to evaluation mode
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    decoded_samples, noise_samples = [], []
    test_metric_values = []  # List to store PSNR values for each image
    with torch.no_grad():
        for images, (degradations, depths) in tqdm(dataloader, desc='Test'):
            images = images.to(device) * 2 - 1 # mapped to (-1, 1)
            degradations = degradations.to(device)
            depths = depths.to(device)
            

            # The random starting point
            latents = torch.randn_like(degradations).to(device)

            # Initialize an empty lists to store individual grids
            grid_dict = {'samples': [], 'steps': []}
            # Loop through the sampling timesteps
            for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
                # Get the prediction
                noise_pred = unet(latents, t, depths, degradations)

                # Calculate what the updated sample should look like with the scheduler
                scheduler_output = noise_scheduler.step(noise_pred, t, latents)

                # Update latents
                latents = scheduler_output.prev_sample

                # Occasionally add the grid to the list
                if i % 10 == 0 or i == len(noise_scheduler.timesteps) - 1:
                    grid = torchvision.utils.make_grid(latents, nrow=4).permute(1, 2, 0)
                    grid_dict['samples'].append(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
                    grid_dict['steps'].append(i)

            decoded = vae.decode(latents).sample

            # Compute PSNR for each image in the batch and store it
            for orig, recon in zip(images, decoded):
                test_metric_value = test_metric(orig, recon)
                test_metric_values.append(test_metric_value)

            noise_samples.append(grid_dict)
            decoded_samples.append(decoded)
    
    # Calculate mean PSNR for all test batches
    mean_test_metric = sum(test_metric_values) / len(test_metric_values)
    
    if log == "mlflow":
        mlflow.log_metric("mean_test_metric", mean_test_metric, step=epoch)
    elif log == "wandb":
        wandb.log({"mean_test_metric": mean_test_metric, "epoch": epoch})

    return decoded_samples, mean_test_metric, noise_samples
    
