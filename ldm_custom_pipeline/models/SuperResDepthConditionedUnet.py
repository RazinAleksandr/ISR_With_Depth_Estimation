import torch.nn as nn
import torch

from diffusers import UNet2DModel


class SuperResDepthConditionedUnet(nn.Module):
    def __init__(self, sample_size=28, input_channels=3, layers_per_block=2, block_out_channels=(32, 64, 64)):
        super().__init__()

        # Assuming depth map has 1 channel and degraded image has the same channels as the input
        self.model = UNet2DModel(
            sample_size=sample_size,                  # the target image resolution
            in_channels=input_channels * 2 + 1,  # input_channels for image, input_channels for degraded image, and 1 for depth map
            out_channels=input_channels,     # the number of output channels
            layers_per_block=layers_per_block,              # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels, 
            down_block_types=( 
                "DownBlock2D",        
                "AttnDownBlock2D",    
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      
                "UpBlock2D",          
            ),
        )

    def forward(self, x, t, depth_map, degraded_x):
        # Concatenate the depth map, degraded image, and the original image along the channel dimension
        net_input = torch.cat((x, degraded_x, depth_map), 1)  

        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(net_input, t).sample
