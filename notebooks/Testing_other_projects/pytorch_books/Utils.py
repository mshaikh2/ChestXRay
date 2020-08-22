import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def fast_rescale(x, xmin, xmax, smin, smax):
    return smin + ((x - xmin) * (smax - smin)) / (xmax - xmin)