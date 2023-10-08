import mlflow
import torch
from tqdm import tqdm


def train_one_epoch(unet, vae, image_dataloader, cond_dataloader, opt, device, num_train_timesteps, noise_scheduler, loss_fn, pbar):
    unet.train()

    losses = []
    images = next(iter(image_dataloader))
    degradations, depths = next(iter(cond_dataloader))
    for images, (degradations, depths) in tqdm(zip(image_dataloader, cond_dataloader)):
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

        losses.append(loss.item())

        pbar.update(1)

    return sum(losses) / len(losses)

def validate_one_epoch(unet, vae, image_dataloader, cond_dataloader, device, num_train_timesteps, noise_scheduler, loss_fn, pbar):
    unet.eval()  # Set the model to evaluation mode
    
    losses = []
    with torch.no_grad():
        for images, (degradations, depths) in tqdm(zip(image_dataloader, cond_dataloader)):
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
            
            pbar.set_description(f"Validation Loss: {loss.item():.4f}")
            pbar.update(1)
    
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
        opt,
        noise_scheduler,
        loss_fn,
        mlflow_log=False):
    
    train_losses = []
    val_losses = []
    total_iterations = len(train_image_dataloader) + len(val_image_dataloader)
    
    for epoch in range(1, n_epochs+1):    
        pbar = tqdm(total=total_iterations, desc=f"Epoch {epoch}/{n_epochs}")

        # Training
        train_loss = train_one_epoch(unet, vae, train_image_dataloader, train_cond_dataloader, opt, device, num_train_timesteps, noise_scheduler, loss_fn, pbar)
        train_losses.append(train_loss)

        # Validation
        val_loss = validate_one_epoch(unet, vae, val_image_dataloader, val_cond_dataloader, device, num_train_timesteps, noise_scheduler, loss_fn, pbar)
        val_losses.append(val_loss)

        if mlflow_log:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        pbar.close()
        print(f"Epoch {epoch}/{n_epochs} => Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(' ')

    return train_losses, val_losses


def test(unet, vae, image_dataloader, cond_dataloader, device, num_train_timesteps, noise_scheduler):
    unet.eval()  # Set the model to evaluation mode
    
    total_iterations = len(image_dataloader)
    pbar = tqdm(total=total_iterations, desc=f"Process test sampes")
    decoded_samples = []
    with torch.no_grad():
        for images, (degradations, depths) in tqdm(zip(image_dataloader, cond_dataloader)):
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

            pbar.update(1)
    
    return decoded_samples
    
    
