import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from data_utils import TestDataset
import os

from model3d import Generator3D

import torchvision.transforms as transforms
from torch.utils.data import DataLoader



transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
])


img = transform(Image.open('/home/gridsan/jmeindl/SRGAN/data/REDS/train_sharp_bicubic/X4/004/00000002.png'))

out_img = ToPILImage()(img.data.cpu())

out_img.save('test.png')