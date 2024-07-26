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


parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_5.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
# IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
])

model = Generator3D(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    # model.load_state_dict(torch.load('test/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load('base_dis3/' + MODEL_NAME, map_location=lambda storage, loc: storage))


# folder_path1 = 'data/REDS/val_REDS4_sharp'
# input_path = 'data/REDS/val_REDS4_sharp_bicubic/X4'
# input_path = 'data/REDS/low_res_val'
# input_path = 'data/REDS/low_res_val_bicubic'

# input_path = 'data/REDS/val_REDS4_sharp_bicubic/X4'
input_path = 'data/VID4/BDx4'

# output_path = 'data/REDS/base_pred_val'

# output_path = 'data/REDS/base_highres_pred_val'
# output_path = 'data/REDS/pred2'

output_path = 'data/VID4/BDx4/base_pred'
# output_path = 'data/VID4/BDx4/new_pred'


dataset = TestDataset(input_path, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for lr_batch, path in loader:
    if torch.cuda.is_available():
        lr_batch = lr_batch.to(device)
    # print(lr_batch)
    upscaled = model(lr_batch)
    # print(upscaled)
    out_img = ToPILImage()(upscaled.squeeze(0).data.cpu())
    subdirectory = '/'.join(path[1][0].split('/')[-2:])
    save_path = os.path.join(output_path, subdirectory)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out_img.save(save_path)

