import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tfms
import torchvision.datasets as dsets

from torch.utils.data import DataLoader
from model import DarkNet

train_transforms = tfms.Compose([
    tfms.RandomResizedCrop(224),
    tfms.RandomHorizontalFlip(),
    tfms.ToTensor(),
    tfms.Normalize([0.485, 0.456, 0.406],
                   [0.229, 0.224, 0.225])
])

test_transforms = tfms.Compose([
    tfms.Resize(256),
    tfms.CenterCrop(224),
    tfms.ToTensor(),
    tfms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
])


train_dataset = dsets.ImageNet('./data/ImageNet', download=True, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)