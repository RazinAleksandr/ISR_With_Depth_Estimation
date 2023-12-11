import torch.nn as nn
import torch


class DepthMapCNN(nn.Module):
    def __init__(self, num_in_ch, embed_dim):
        super(DepthMapCNN, self).__init__()
        
        self.conv_depth = nn.Sequential(
            nn.Conv2d(num_in_ch, num_in_ch, 3, 1, 1),
            nn.BatchNorm2d(num_in_ch),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(inplace=True),
        )
        
        
    def forward(self, x, x_depth):
        # out = self.conv_depth(torch.cat((x, x_depth), dim = 1))
        out = self.conv_depth(x)
        # out = self.conv_depth(x_depth)
        out = out * x_depth
        return out
