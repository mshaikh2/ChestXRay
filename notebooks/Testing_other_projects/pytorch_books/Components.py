import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_op = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1))
        self.key_op = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1))
        self.value_op = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))
        self.safm_op = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        n, c, h, w = x.shape
        query = self.query_op(x).view(n, -1, h*w).permute(0, 2, 1)
        key = self.key_op(x).view(n, -1, h*w)
        value = self.value_op(x).view(n, -1, h*w)
        attn_map = F.softmax(torch.bmm(query, key), dim=-1)
        safm = self.safm_op(torch.bmm(attn_map, value).view(n, c, h, w))
        return x + self.gamma*safm, attn_map