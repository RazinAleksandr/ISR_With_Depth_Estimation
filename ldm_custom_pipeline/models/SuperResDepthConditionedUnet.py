# Third-party libraries
import torch.nn as nn
import torch

from diffusers import UNet2DModel


class SuperResDepthConditionedUnet(nn.Module):
    """
    A U-Net based architecture conditioned on super-resolution depth maps. This model takes in an original image,
    a degraded version of the image, and a depth map, and produces a super-resolved image.
    
    Attributes:
    model (nn.Module): The underlying U-Net architecture used for the super-resolution task.
    """

    def __init__(self, sample_size=28, input_channels=3, layers_per_block=2, block_out_channels=(32, 64, 64)):
        """
        Initializes the SuperResDepthConditionedUnet model.
        
        Parameters:
        sample_size (int): The target resolution of the image. Default is 28.
        input_channels (int): The number of channels in the input image. Default is 3 (RGB).
        layers_per_block (int): The number of ResNet layers used per U-Net block. Default is 2.
        block_out_channels (tuple of int): Specifies the number of output channels for each U-Net block. Default is (32, 64, 64).
        """
        super().__init__()

        # Assuming depth map has 1 channel and degraded image has the same channels as the input
        self.model = UNet2DModel(
            sample_size=sample_size,                # the target image resolution
            in_channels=input_channels * 2 + 1,     # Input channels: image + degraded image + depth map
            out_channels=input_channels,            # the number of output channels
            layers_per_block=layers_per_block,      # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels,
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"), 
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
        )

    # def forward(self, x, t, depth_map, degraded_x):
    #     """
    #     Forward pass of the SuperResDepthConditionedUnet model.
        
    #     Parameters:
    #     x (torch.Tensor): The original image tensor.
    #     t (torch.Tensor): The time step tensor for conditional generation.
    #     depth_map (torch.Tensor): The depth map tensor.
    #     degraded_x (torch.Tensor): The degraded version of the image tensor.
        
    #     Returns:
    #     torch.Tensor: The super-resolved image tensor.
    #     """
    #     # Concatenate the depth map, degraded image, and the original image along the channel dimension
    #     net_input = torch.cat((x, degraded_x, depth_map), 1)  

    #     # Feed this to the U-Net alongside the timestep and return the prediction
    #     return self.model(net_input, t).sample
    def forward(self, latents, t):
        """
        Forward pass of the SuperResDepthConditionedUnet model.
        
        Parameters:
        x (torch.Tensor): The original image tensor.
        t (torch.Tensor): The time step tensor for conditional generation.
        depth_map (torch.Tensor): The depth map tensor.
        degraded_x (torch.Tensor): The degraded version of the image tensor.
        
        Returns:
        torch.Tensor: The super-resolved image tensor.
        """

        # Feed this to the U-Net alongside the timestep and return the prediction
        return self.model(latents, t).sample