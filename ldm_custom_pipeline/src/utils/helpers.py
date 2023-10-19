import mlflow
import wandb
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


# Replace string class names with actual classes using the mapping
def str_to_class(exp_config, CLASS_MAPPING):
    for dataset in ['train_datasets', 'val_datasets', 'test_datasets']:    
        exp_config[dataset]['image_dataset'] = CLASS_MAPPING[exp_config[dataset]['image_dataset']]
        exp_config[dataset]['cond_dataset'] = CLASS_MAPPING[exp_config[dataset]['cond_dataset']]
        exp_config[dataset]['transform'] = CLASS_MAPPING[exp_config[dataset]['transform']]
        exp_config[dataset]['degradation'] = CLASS_MAPPING[exp_config[dataset]['degradation']]
        
    exp_config['model']['pretrain_pipeline'] = CLASS_MAPPING[exp_config['model']['pretrain_pipeline']]
    exp_config['model']['noise_scheduler'] = CLASS_MAPPING[exp_config['model']['noise_scheduler']]
    exp_config['model']['loss'] = CLASS_MAPPING[exp_config['model']['loss']]
    exp_config['model']['test_metric'] = CLASS_MAPPING[exp_config['model']['test_metric']]

    return exp_config


def save_and_log_images(decoded_samples, noise_samples, epoch, logdir, log=False):
    for idx, (image, grid_dict) in enumerate(zip(decoded_samples, noise_samples)):
        # Save the image to a file
        file_name = f"reconstructed_epoch_{epoch}_sample_{idx}.png"
        file_path = os.path.join(logdir, file_name)
        vutils.save_image(image, file_path)

        n = len(grid_dict['samples'])
        fig, axs = plt.subplots(n, 1, figsize=(12, 5))
        for i in range(len(grid_dict['samples'])):
            axs[i].imshow(grid_dict['samples'][i])
            axs[i].set_title(f"Current x (step {grid_dict['steps'][i]})")
        plt.tight_layout()

        # Save the concatenated image as a file
        concatenated_file_name = f"unet_epoch_{epoch}_sample_{idx}.png"
        concatenated_file_path = os.path.join(logdir, concatenated_file_name)
        plt.savefig(concatenated_file_path)

        # Log the concatenated image file to MLflow
        if log == "mlflow":
            mlflow.log_artifact(file_path)
            mlflow.log_artifact(concatenated_file_path)
        elif log == "wandb":
            wandb.log({"reconstructed_images": [wandb.Image(image, caption=file_name)], 
                    #    "concatenated_images": [wandb.Image(concatenated_image, caption=concatenated_file_name)]})
                       "concatenated_images": [wandb.Image(fig)]})


def save_checkpoint(epoch, model, optimizer, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch, model, optimizer, loss
    else:
        print("No checkpoint found.")
        return None, model, optimizer, None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
