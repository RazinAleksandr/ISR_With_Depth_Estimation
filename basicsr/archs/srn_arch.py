import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from basicsr.utils.registry import ARCH_REGISTRY


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4     
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        c1 = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(c1)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)
        return x * m

class ResidualBlock_ESA(nn.Module):
    '''
    ---Conv-ReLU-Conv-ESA +-
    '''
    def __init__(self, nf=32):
        super(ResidualBlock_ESA, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.ESA = ESA(nf, nn.Conv2d)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = (self.conv1(x))
        out = self.lrelu(out)
        out = (self.conv2(out))
        out = self.ESA(out)
        return out 

# @ARCH_REGISTRY.register()
class SRN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=4, num_out_ch=3, upscale=4):
        super(SRN, self).__init__()
        basic_block = functools.partial(ResidualBlock_ESA, nf=num_feat)
        self.recon_trunk = make_layer(basic_block, num_block)
        self.head = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.tail = nn.Conv2d(num_feat, num_out_ch * upscale * upscale, kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        fea = self.head(x)
        out = fea
        layer_names = self.recon_trunk._modules.keys()
        for layer_name in layer_names:
            fea = self.recon_trunk._modules[layer_name](fea)
        out = fea + out
        out = self.pixel_shuffle(self.tail(out))
        return out


if __name__== '__main__':
    #############Test Model Complexity #############
    model = SRN()
    # print(model)
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    from utility.model_summary import get_model_flops, get_model_activation

    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))