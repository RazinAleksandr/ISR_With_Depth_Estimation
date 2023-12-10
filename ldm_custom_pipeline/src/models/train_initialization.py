import gc

# Third-party libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import LDMSuperResolutionPipeline, DDIMScheduler

# Local application/modules
from models.SuperResDepthConditionedUnet import SuperResDepthConditionedUnet
from src.data.datasets import ImageDataset, ConditionDataset, CombinedDataset
from src.logging.CustomLogger import CustomLogger
from src.metrics.PSNR import PSNR
from src.utils.degradations import BSRDegradation
from src.utils.helpers import create_folder_if_not_exists
from src.utils.transforms import BaseTransform
from src.constants.types import Optional, Any, Dict, Tuple


def initialize_parameters(n_epochs=10, val_step=100, model_name='model.pth', seed=17) -> Dict[str, Any]:
    """
    Initialize parameters for the training process.
    
    :param n_epochs: Number of epochs to train for.
    :param val_step: Number of iterations after which validation is performed.
    :param model_name: Name of the model. Used for saving the model checkpoint.
    :param seed: Random seed for reproducibility.
    
    :return: A dictionary containing the initialized parameters.
    """
    config = {
        'n_epochs': n_epochs,
        'val_step': val_step,
        'model_name': model_name,
        'seed': seed,
    }
    return config


def initialize_datasets_and_dataloaders(
        image_dir:                          str,
        depth_dir:                          str,
        batch_size:                         int = 4,
        shuffle:                            bool = True,
        transform:                          Optional[Any] = BaseTransform,
        degradation:                        Optional[Any] = BSRDegradation,
        image_dataset:                      Optional[Any] = ImageDataset,
        cond_dataset:                       Optional[Any] = ConditionDataset,
        image_transform_params:             Dict[str, Any] = {'size': 128, 'resize': True},
        cond_transform_params:              Dict[str, Any] = {'size': 32, 'resize': True},
        degradation_params:                 Dict[str, Any] = {}
    ) ->                                    DataLoader:
    """
    Initialize datasets and dataloaders for training and evaluation.

    :param image_dir: Directory path where the image data is stored.
    :param depth_dir: Directory path where the depth data is stored.
    :param batch_size: Number of samples per batch during training/evaluation.
    :param shuffle: Whether to shuffle the dataset before creating batches.
    :param transform: Transformation class/method to apply to the datasets.
    :param degradation: Degradation class/method to degrade the image quality.
    :param image_dataset: Class used to create an image dataset.
    :param cond_dataset: Class used to create a condition dataset based on image and depth data.
    :param image_transform_params: Dictionary of parameters to be passed to the transform for the image dataset.
    :param cond_transform_params: Dictionary of parameters to be passed to the transform for the condition dataset.
    :param degradation_params: Dictionary of parameters to be passed to the degradation method.

    :return: A DataLoader instance containing the combined datasets for both image and condition.
    """

    image_dataset = image_dataset(image_dir=image_dir, transform=transform(**image_transform_params))
    cond_dataset = cond_dataset(image_dir=image_dir, depth_dir=depth_dir, transform=transform(**cond_transform_params), degradation=degradation(**degradation_params))
    
    combined_dataset = CombinedDataset(image_dataset, cond_dataset)
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)

    return combined_dataloader


def initialize_model_and_optimizer(
        device:                         str = 'cpu',
        lr:                             float = 1e-3,
        lr_scheduler:                   Optional[Any] = None, # CosineAnnealingLR
        lr_scheduler_params:            Dict[str, Any] = {
            'T_max':                    10,
            'eta_min':                  1e-6,
            'last_epoch':               -1,
            'verbose':                  False,
        },
        unet_model_params:              Dict[str, Any] = {
            'sample_size':              32,
            'input_channels':           3,
            'layers_per_block':         2,
            'block_out_channels':       (32, 64, 64)
        },
        pretrain_pipeline:              Optional[Any] = LDMSuperResolutionPipeline,
        pretrain_model_id:              str = "CompVis/ldm-super-resolution-4x-openimages",
        noise_scheduler:                Optional[Any] = DDIMScheduler,
        scheduler_train_params:         Dict[str, Any] = {
            'num_train_timesteps':      10,
            'beta_schedule':            'squaredcos_cap_v2'
        },
        loss:                           Optional[Any] = nn.MSELoss,
        test_metric:                    Optional[Any] = PSNR,
        **kwargs
    ) ->                                Tuple:
    """
    Initialize model components including pre-trained models, optimizers, schedulers, etc.

    :param device: Device on which the models should run. Options include 'cpu', 'cuda', etc.
    :param lr: Learning rate to be used for the optimizer.
    :param lr_scheduler: Learning rate scheduler (epoch steps).
    :param lr_scheduler_params: Learning rate scheduler parameters.
    :param unet_model_params: Dictionary containing parameters for initializing the U-Net model.
    :param pretrain_pipeline: Pipeline used for initializing pre-trained models.
    :param pretrain_model_id: Identifier for the pre-trained model.
    :param noise_scheduler: Scheduler class/method for handling noise during training.
    :param scheduler_train_params: Dictionary containing parameters for the noise scheduler.
    :param loss: Loss function class to use during training.
    :param test_metric: Metric class to use during testing.
    :param kwargs: Additional keyword arguments for future extensions or model parameters.

    :return: A tuple containing initialized models (VAE, U-Net), noise scheduler, loss function, test metric, optimizer and lr_scheduler.
    """
    
    # Initialize VAE
    sr_pipe = pretrain_pipeline.from_pretrained(pretrain_model_id)
    vae = sr_pipe.vqvae.to(device)
    
    # Clean cache
    del sr_pipe
    gc.collect()

    # Initialize conditional U-Net
    unet = SuperResDepthConditionedUnet(**unet_model_params).to(device)
    
    # Create a noise scheduler
    noise_scheduler = noise_scheduler(**scheduler_train_params)
    
    # Initialize loss function
    loss_fn = loss()
    
    # Test metric
    test_metric = test_metric
    
    # Initialize optimizer
    opt = torch.optim.Adam(unet.parameters(), lr=lr)

    if lr_scheduler:
        lr_scheduler = lr_scheduler(optimizer=opt, **lr_scheduler_params)
    
    # return vae, unet, noise_scheduler, loss_fn, test_metric, opt, lr_scheduler, sr_pipe
    return vae, unet, noise_scheduler, loss_fn, test_metric, opt, lr_scheduler


def initialize_logfolders(logdir_path: str, experiment_name: str) -> str:
    """
    Initialize logging directories for saving models and test samples.
    
    :param logdir_path: Base directory for all logging.
    :param experiment_name: Name of the experiment for the current run.
    
    :return: Path to the specific log directory for the current experiment.
    """
    logdir = f'{logdir_path}/{experiment_name}'

    create_folder_if_not_exists(f'{logdir}/models')
    create_folder_if_not_exists(f'{logdir}/test_samples')
    
    return logdir


def initialize_logger(log_config: Dict[str, Any]) -> CustomLogger:
    """
    Initialize the CustomLogger with the specified backend.
    
    :param exp_config: Configuration dictionary with logging details.
    
    :return: An instance of CustomLogger.
    """
    # Extract logger and experiment_name or use None if not available
    backend = log_config.get('logger', None)
    if backend and backend.lower() == "none":
        return None

    project_name = log_config.get('project_name', None)
    experiment_name = log_config.get('experiment_name', None)

    return CustomLogger(backend=backend, project_name=project_name, exp_name=experiment_name)

