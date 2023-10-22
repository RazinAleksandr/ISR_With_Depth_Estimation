import torch

from src.constants.constants import PSNR_MAX_PIXEL


def PSNR(target, prediction):
    target =  target * 2 - 1 # mapped to (-1, 1) 
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = PSNR_MAX_PIXEL
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

