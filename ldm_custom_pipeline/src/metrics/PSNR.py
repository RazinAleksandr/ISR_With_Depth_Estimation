import torch


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Since the images are between -1 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

