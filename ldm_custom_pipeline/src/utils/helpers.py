import mlflow
import torchvision.utils as vutils
import os


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

    return exp_config


def save_and_log_images(decoded_samples, epoch, logdir, mlflow_log=False):
    for idx, image in enumerate(decoded_samples):
        # Save the image to a file
        file_name = f"reconstructed_epoch_{epoch}_sample_{idx}.png"
        file_path = os.path.join(logdir, file_name)
        vutils.save_image(image, file_path)

        # Log the file to MLflow
        if mlflow_log:
            mlflow.log_artifact(file_path)
