import torch.nn as nn
from helper import ResBlock, AttnBlock, DownSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128,128, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(3, channels[0], 3, 1, 1)]#args.image_channels
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResBlock(channels[-1], channels[-1]))
        layers.append(AttnBlock(channels[-1]))
        layers.append(ResBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], 512, 3, 1, 1))#args.latent_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)