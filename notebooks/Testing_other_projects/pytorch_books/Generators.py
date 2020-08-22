import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import Components

class cSAWGANGeneratorV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cSAWGANGeneratorV1, self).__init__()
        # (N, in_channels, 1, 1)
        self.l1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=1024, kernel_size=4, stride=1, padding=0)),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.1)
        )

        # (N, 1024, 4, 4)
        self.l2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1)
        )

        # (N, 512, 8, 8)
        self.l3 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1)
        )

        self.attn = Components.SelfAttention(in_channels=256)

        # (N, 256, 16, 16)
        self.l4 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=out_channels, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()
        )
        # (N, out_channels, 32, 32)
    
    def forward(self, z, y):
        res = torch.cat([z, y], dim=1)
        res = res.view(res.shape[0], res.shape[1], 1, 1)
        res = self.l1(res)
        res = self.l2(res)
        res = self.l3(res)
        res, attn_map = self.attn(res)
        res = self.l4(res)
        #return res, attn_map
        return res