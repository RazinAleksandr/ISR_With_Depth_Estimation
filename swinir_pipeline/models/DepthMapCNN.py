import torch.nn as nn


class DepthMapCNN(nn.Module):
    def __init__(self, num_in_ch, embed_dim):
        super(DepthMapCNN, self).__init__()

        self.conv_depth = nn.Sequential(
            nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(inplace=True)
        )
        
        
    def forward(self, x, x_depth):
        x_depth_conv = self.conv_depth(x_depth)
        x = x * x_depth_conv
        
        return x