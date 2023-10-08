import click
import yaml

import mlflow
import wandb

# Importing necessary modules and functions
from src.utils.helpers import str_to_class
from src.constants.classes import CLASS_MAPPING
from src.models.train_initialization import initialize_parameters, initialize_datasets_and_dataloaders, initialize_model_and_optimizer
from src.models.train_utils import fit


@click.command()
@click.option('--config_path', default='config.yaml', help='Path to the configuration YAML file.')
def main(config_path):
    # Load the configuration file
    with open(config_path, 'r') as stream:
        exp_config = yaml.safe_load(stream)

    # substitute class names from config to classes
    exp_config = str_to_class(exp_config, CLASS_MAPPING)

    # Now you can use the configuration values
    initial_params = initialize_parameters(**exp_config['initialization'])
    train_image_dataloader, train_cond_dataloader = initialize_datasets_and_dataloaders(**exp_config['train_datasets'])
    val_image_dataloader, val_cond_dataloader = initialize_datasets_and_dataloaders(**exp_config['val_datasets'])
    test_image_dataloader, test_cond_dataloader = initialize_datasets_and_dataloaders(**exp_config['test_datasets'])
    vae, unet, noise_scheduler, loss_fn, opt = initialize_model_and_optimizer(**exp_config['model'])
    
    # get some train params
    device = exp_config['model']['device']
    num_train_timesteps = exp_config['model']['num_train_timesteps']
    n_epochs = initial_params['n_epochs']
    exp_name = exp_config['logging']['experiment_name']
    logdir = exp_config['logging']['logdir']
    log = exp_config['logging']['log']
    val_step = initial_params['val_step']
    

    # logging
    if log == "mlflow":
        # mlflow.start_run(experiment_id=exp_name)
        mlflow.start_run()
        mlflow.log_params(initial_params) # subst. to exp_config
    elif log == "wandb":
        # Initialize wandb
        wandb.init(project="ldm_conditioned", name=exp_name)  # Replace "your_project_name" with your W&B project name
        wandb.config.update(exp_config)  # Log the experiment configuration to W&B
        wandb.watch([unet, vae], log="all")  # Log model gradients and parameters


    # run train/val loop
    train_loss_history, val_loss_history = fit(
        n_epochs, 
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
        log,
        val_step
        )

    if log == "mlflow":
        mlflow.finish()
    elif log == "wandb":
        wandb.finish()

if __name__ == "__main__":
    main()
