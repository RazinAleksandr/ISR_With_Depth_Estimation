# Standard libraries
import click
import yaml

# Third-party libraries
import mlflow
import wandb

# Local modules and functions
from src.constants.classes import CLASS_MAPPING
from src.utils.helpers import str_to_class, set_seed
from src.models.model_utils import load_checkpoint
from src.models.train_initialization import (initialize_parameters, 
                                             initialize_logfolders, 
                                             initialize_datasets_and_dataloaders, 
                                             initialize_model_and_optimizer)
from src.models.train_utils import fit



@click.command()
@click.option('--config_path', default='config.yaml', help='Path to the configuration YAML file.')
@click.option('--checkpoint_resume', default=None, help='Path to the checkpoint.')
def main(config_path: str, checkpoint_resume: str = None) -> None:
    """
    Main entry point of the script. It initializes the training process given a configuration and optionally
    resumes from a provided checkpoint.

    :param config_path: Path to the YAML configuration file.
    :param checkpoint_resume: Optional path to a checkpoint to resume training.
    """

    # Load the configuration file
    with open(config_path, 'r') as stream:
        exp_config = yaml.safe_load(stream)

    # Substitute string class names from config to actual classes
    exp_config = str_to_class(exp_config, CLASS_MAPPING)

    # Initialize parameters from the configuration
    initial_params = initialize_parameters(**exp_config['initialization'])
    
    # Create directories for logging outputs
    logdir = initialize_logfolders(exp_config['logging']['logdir'], exp_config['logging']['experiment_name'])

    # Set random seed for reproducibility
    set_seed(initial_params['seed'])

    # Initialize data loaders
    train_dataloader = initialize_datasets_and_dataloaders(**exp_config['train_datasets'])
    val_dataloader = initialize_datasets_and_dataloaders(**exp_config['val_datasets'])
    test_dataloader = initialize_datasets_and_dataloaders(**exp_config['test_datasets'])

    # Initialize model, optimizer, and other necessary components
    vae, unet, noise_scheduler, loss_fn, test_metric, opt = initialize_model_and_optimizer(**exp_config['model'])
    
    # Extract training parameters
    device = exp_config['model']['device']
    num_train_timesteps = exp_config['model']['scheduler_train_params']['num_train_timesteps']
    n_epochs = initial_params['n_epochs']
    exp_name = exp_config['logging']['experiment_name']
    log = exp_config['logging']['log']
    val_step = initial_params['val_step']
    model_name = initial_params['model_name']
    num_inference_steps = exp_config['model']['scheduler_inference_params']['num_inference_steps']
    
    # If a checkpoint is provided, load model and optimizer state
    start_epoch = 0
    if checkpoint_resume:
        start_epoch, unet, opt, _ = load_checkpoint(unet, opt, checkpoint_resume)
        print(f'Resume train from {start_epoch}, checkpoint_path: {checkpoint_resume}')

    # Initialize logging frameworks
    if log == "mlflow":
        mlflow.start_run()
        mlflow.log_params(initial_params)
    elif log == "wandb":
        wandb.init(project="ldm_conditioned", name=exp_name)
        wandb.config.update(exp_config)
        wandb.watch([unet, vae], log="all")


    # Start training and validation loop
    fit(
        n_epochs, 
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
        log,
        model_name,
        val_step,
        start_epoch,
        test_metric,
        num_inference_steps
    )

    # Close logging sessions
    if log == "mlflow":
        mlflow.finish()
    elif log == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
