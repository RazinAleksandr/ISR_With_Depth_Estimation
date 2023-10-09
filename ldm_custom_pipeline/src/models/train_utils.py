import mlflow
import wandb

import torch
from tqdm import tqdm

from src.utils.helpers import save_and_log_images

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

def validate(unet, vae, image_dataloader, cond_dataloader, device, num_train_timesteps, noise_scheduler, loss_fn, epoch, log=None):
    unet.eval()  # Set the model to evaluation mode

    losses = []
    with torch.no_grad():
        for images, (degradations, depths) in tqdm(zip(image_dataloader, cond_dataloader), desc="Validation"):
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
        train_image_dataloader, 
        train_cond_dataloader, 
        val_image_dataloader, 
        val_cond_dataloader,
        test_image_dataloader,
        test_cond_dataloader,
        opt,
        noise_scheduler,
        loss_fn,
        logdir,
        log=None,
        val_step=400):

    train_loss_history, val_loss_history = [], []

    total_iterations = len(train_image_dataloader)
    iteration = 0  # Initialize iteration counter
    for epoch in range(1, n_epochs+1):

        epoch_train_losses, epoch_val_losses = [], []
        pbar = tqdm(total=total_iterations, desc=f"Epoch {epoch}/{n_epochs}")
        for images, (degradations, depths) in tqdm(zip(train_image_dataloader, train_cond_dataloader)):
            # Training
            unet.train()
            train_loss = train_one_batch(unet, vae, images, degradations, depths, opt, device, num_train_timesteps, noise_scheduler, loss_fn, epoch, log)
            
            iteration += 1  # Increment iteration counter
            pbar.update(1)

            # Check if it's time for validation
            if iteration % val_step == 0:
                val_loss = validate(unet, vae, val_image_dataloader, val_cond_dataloader, device, num_train_timesteps, noise_scheduler, loss_fn, epoch, log)
                
                if log == "mlflow":
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                elif log == "wandb":
                    wandb.log({"val_loss": val_loss, "epoch": epoch})
                
                epoch_val_losses.append(val_loss)

            epoch_train_losses.append(train_loss)

        # Test and save images after each epoch
        test_prediction = test(unet, vae, test_image_dataloader, test_cond_dataloader, device, num_train_timesteps, noise_scheduler)
        save_and_log_images(test_prediction, epoch, logdir, log)


        pbar.close()

        epoch_train_losses = sum(epoch_train_losses) / len(epoch_train_losses)
        epoch_val_losses = sum(epoch_val_losses) / len(epoch_val_losses)

        train_loss_history.append(epoch_train_losses)
        val_loss_history.append(epoch_val_losses)

        print(f"Epoch {epoch}/{n_epochs} => Train Loss: {epoch_train_losses:.4f}, Validation Loss: {epoch_val_losses:.4f}")
        print(' ')
    
    return train_loss_history, val_loss_history


def test(unet, vae, image_dataloader, cond_dataloader, device, num_train_timesteps, noise_scheduler):
    unet.eval()  # Set the model to evaluation mode
    
    decoded_samples = []
    with torch.no_grad():
        for images, (degradations, depths) in tqdm(zip(image_dataloader, cond_dataloader), desc='Test'):
            images = images.to(device) * 2 - 1
            degradations = degradations.to(device)
            depths = depths.to(device)
            
            latents = 0.18215 * vae.encode(images).latents
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, num_train_timesteps-1, (latents.shape[0],)).long().to(device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            pred = unet(noisy_latents, timesteps, depths, degradations)

            decoded = vae.decode(pred / 0.18215).sample
            decoded_samples.append(decoded)
    
    return decoded_samples
    
    
