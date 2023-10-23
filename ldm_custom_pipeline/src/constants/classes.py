from diffusers import LDMSuperResolutionPipeline, DDIMScheduler
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR, MultiStepLR

from src.data.datasets import ImageDataset, ConditionDataset
from src.utils.transforms import BaseTransform
from src.utils.degradations import BSRDegradation
from src.metrics.PSNR import PSNR


# Mapping from string names to actual classes
CLASS_MAPPING = {
    "ImageDataset": ImageDataset,
    "ConditionDataset": ConditionDataset,
    "BaseTransform": BaseTransform,
    "BSRDegradation": BSRDegradation,
    "LDMSuperResolutionPipeline": LDMSuperResolutionPipeline,
    "DDIMScheduler": DDIMScheduler,
    "MSE": nn.MSELoss,
    "PSNR": PSNR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ExponentialLR": ExponentialLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    None: None
}