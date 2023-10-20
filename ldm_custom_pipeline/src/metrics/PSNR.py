import torch


def PSNR(target, prediction):
    target =  target * 2 - 1 # mapped to (-1, 1) 
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Since the images are between -1 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

