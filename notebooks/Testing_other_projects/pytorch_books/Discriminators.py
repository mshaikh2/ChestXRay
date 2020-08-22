import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import Components

class cSAWGANDiscriminatorV1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(cSAWGANDiscriminatorV1, self).__init__()
        # (N, in_channels, 32, 32)
        self.l1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2)
        )

        self.attn = Components.SelfAttention(in_channels=256)

        # (N, 256, 16, 16)
        self.l2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)
        )

        # (N, 512, 32, 32)
        self.l3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2)
        )

        # (N, 1024, 4, 4)
        self.l4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=1, padding=0)),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2)
        )

        # (N, 1024, 1, 1)
        self.l5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_features=1024+num_classes, out_features=1))
        )


    def forward(self, x, y):
        res = self.l1(x)
        res, attn_map = self.attn(res)
        res = self.l2(res)
        res = self.l3(res)
        res = self.l4(res)
        res = self.l5(torch.cat([res.view(-1, 1024), y], dim=1))
        #return res, attn_map
        return res