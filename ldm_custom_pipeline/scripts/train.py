# Standard libraries
import click
import yaml

# Local modules and functions
from src.constants.classes import CLASS_MAPPING
from src.utils.helpers import str_to_class, set_seed
from src.models.model_utils import load_model_and_optimizer
from src.models.train_initialization import (initialize_parameters, 
                                             initialize_logfolders, 
                                             initialize_datasets_and_dataloaders, 
                                             initialize_model_and_optimizer,
                                             initialize_logger)
from src.models.train_utils import fit


def initialization(logdir_path, experiment_name, **params):
    # Initialize parameters from the configuration
    initial_params = initialize_parameters(**params)
    
    # Create directories for logging outputs
    logdir = initialize_logfolders(logdir_path, experiment_name)

    return initial_params, logdir

def get_loaders(**params):
    assert len(params.keys()) == 3, "train_datasets, val_datasets, test_datasets"
    
    dataloaders = []
    for dataset in params.keys():
        dataloaders.append( initialize_datasets_and_dataloaders(**params[dataset]) )

    return dataloaders


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

    # Initialize parameters, log folders
    initial_params, logdir = initialization(exp_config['logging']['logdir'], exp_config['logging']['experiment_name'], **exp_config['initialization'])

    # Set random seed for reproducibility
    set_seed(initial_params['seed'])

    # Initialize dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_loaders(**exp_config["datasets"])

    # Initialize model, optimizer, and other necessary components
    # vae, unet, noise_scheduler, loss_fn, test_metric, opt, lr_scheduler, _ = initialize_model_and_optimizer(**exp_config['model'])
    vae, unet, noise_scheduler, loss_fn, test_metric, opt, lr_scheduler = initialize_model_and_optimizer(**exp_config['model'])
    
    # Initialize logger
    logger = initialize_logger(exp_config['logging'])
    
    # If checkpoint path is provided -> resume train with loaded parameters/ weights and etc.
    start_epoch, unet, opt, lr_scheduler = load_model_and_optimizer(unet, opt, lr_scheduler, checkpoint_resume)

    # Log initial parameters of experiment
    if logger:
        logger.log_params(initial_params)

    # Extract training parameters
    device = exp_config['model']['device']
    num_train_timesteps = exp_config['model']['scheduler_train_params']['num_train_timesteps']
    n_epochs = initial_params['n_epochs']
    val_step = initial_params['val_step']
    model_name = initial_params['model_name']
    num_inference_steps = exp_config['model']['scheduler_inference_params']['num_inference_steps']
    lr_scheduler_start_epoch = exp_config['model']['lr_scheduler_start_epoch']

    # Start training and validation loop
    fit(n_epochs                    =               n_epochs, 
        device                      =                 device, 
        num_train_timesteps         =    num_train_timesteps, 
        unet                        =                   unet, 
        vae                         =                    vae, 
        train_dataloader            =       train_dataloader,
        val_dataloader              =         val_dataloader,
        test_dataloader             =        test_dataloader,
        opt                         =                    opt,
        noise_scheduler             =        noise_scheduler,
        loss_fn                     =                loss_fn,
        logdir                      =                 logdir,
        logger                      =                 logger,
        model_name                  =             model_name,
        val_step                    =               val_step,
        start_epoch                 =            start_epoch,
        test_metric                 =            test_metric,
        num_inference_steps         =    num_inference_steps,
        lr_scheduler                =           lr_scheduler,
        lr_scheduler_start_epoch    = lr_scheduler_start_epoch)

    # Finish logging
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
