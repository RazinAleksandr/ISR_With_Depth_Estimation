from diffusers import LDMSuperResolutionPipeline, DDIMScheduler
import torch.nn as nn

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
    "PSNR": PSNR
}