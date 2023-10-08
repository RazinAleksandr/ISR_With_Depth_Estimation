import click
import yaml

import mlflow

# Importing necessary modules and functions
from src.utils.helpers import str_to_class, save_and_log_images
from src.constants.classes import CLASS_MAPPING
from src.models.train_initialization import initialize_parameters, initialize_datasets_and_dataloaders, initialize_model_and_optimizer
from src.models.train_utils import fit, test


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
    mlflow_log = exp_config['logging']['mlflow_log']
    

    # logging
    # mlflow.start_run(experiment_id=exp_name)
    mlflow.start_run()

    mlflow.log_params(initial_params) # subst. to exp_config

    # run train/val loop
    train_losses, val_losses = fit(
        n_epochs, 
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
        mlflow_log
        )
    
    # run test reconstruction
    test_prediction = test(
        unet, 
        vae, 
        test_image_dataloader, 
        test_cond_dataloader, 
        device, 
        num_train_timesteps, 
        noise_scheduler
        )
    
    save_and_log_images(test_prediction, n_epochs, logdir, mlflow_log)


if __name__ == "__main__":
    main()
