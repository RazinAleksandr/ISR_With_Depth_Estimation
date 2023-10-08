from typing import Optional, Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import LDMSuperResolutionPipeline, DDIMScheduler

from src.data.datasets import ImageDataset, ConditionDataset
from src.utils.transforms import BaseTransform
from src.utils.degradations import BSRDegradation
from models.SuperResDepthConditionedUnet import SuperResDepthConditionedUnet


def initialize_parameters(latent_size=32, n_epochs=10, val_step=100):
    config = {
        'latent_size': latent_size,
        'n_epochs': n_epochs,
        'val_step': val_step
    }
    return config

def initialize_datasets_and_dataloaders(image_dir, 
                                        depth_dir,
                                        batch_size=4, 
                                        shuffle=True,
                                        transform: Optional[Any] = BaseTransform,
                                        degradation: Optional[Any] = BSRDegradation,
                                        image_dataset: Optional[Any] = ImageDataset,
                                        cond_dataset: Optional[Any] = ConditionDataset,
                                        image_transform_params: Dict = {'size': 128, 'resize': True},
                                        cond_transform_params: Dict = {'size': 32, 'resize': True},
                                        degradation_params: Dict = {}):

    image_dataset = image_dataset(image_dir=image_dir, transform=transform(**image_transform_params))
    cond_dataset = cond_dataset(image_dir=image_dir, depth_dir=depth_dir, transform=transform(**cond_transform_params), degradation=degradation(**degradation_params))
    
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle)
    cond_dataloader = DataLoader(cond_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return image_dataloader, cond_dataloader

def initialize_model_and_optimizer(device='cpu',
                                   lr=1e-3,
                                   num_train_timesteps=10,
                                   unet_model_params: Dict = {
                                       'sample_size': 32, 
                                       'input_channels': 3, 
                                       'layers_per_block':2, 
                                       'block_out_channels': (32, 64, 64)},
                                   pretrain_pipeline: Optional[Any] = LDMSuperResolutionPipeline,
                                   pretrain_model_id: str = "CompVis/ldm-super-resolution-4x-openimages",
                                   noise_scheduler: Optional[Any] = DDIMScheduler,
                                   loss: Optional[Any] = nn.MSELoss):

    
    # init vae  
    sr_pipe = pretrain_pipeline.from_pretrained(pretrain_model_id)
    vae = sr_pipe.vqvae.to(device)
    # init cond unet
    unet = SuperResDepthConditionedUnet(**unet_model_params).to(device)
    # Create a scheduler
    noise_scheduler = noise_scheduler(num_train_timesteps=num_train_timesteps, beta_schedule='squaredcos_cap_v2')
    # Our loss finction
    loss_fn = loss()
    # opt
    opt = torch.optim.Adam(unet.parameters(), lr=lr)
    
    return vae, unet, noise_scheduler, loss_fn, opt