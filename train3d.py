import argparse
import os
from math import log10

from PIL import Image
import time

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from tqdm import tqdm

import torchvision.transforms as transforms

# Define a transform to convert PIL images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
])

import pytorch_ssim
# from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from data_utils import ImageDataset3D
from loss import GeneratorLoss
from model3d import Generator3D, Discriminator3D
from model import Discriminator
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F



parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')


if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # folder_path1 = 'data/REDS/val_REDS4_sharp'
    # folder_path2 = 'data/REDS/val_REDS4_sharp_bicubic/X4'
    folder_path1 = 'data/REDS/train_sharp_bicubic/X4'
    folder_path2 = 'data/REDS/low_res_train_bicubic'

    directories1 = [d for d in os.listdir(folder_path1) if os.path.isdir(os.path.join(folder_path1, d))]
    directories2 = [d for d in os.listdir(folder_path2) if os.path.isdir(os.path.join(folder_path2, d))]

    common_directories = set(directories1).intersection(directories2)
    dataset = ImageDataset3D(folder_path1, folder_path2, common_directories, transform)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    netG = Generator3D(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator3D()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    device = 'cuda'

    for epoch in range(1, NUM_EPOCHS + 1):
        step = 0  # Initialize step counter
        total_steps = len(loader)
        epoch_start_time = time.time()
        # train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for lr_batch, hr_batch in loader:
            step += 1
            g_update_first = True
            batch_size = hr_batch.shape[0]
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            if torch.cuda.is_available():
                hr_batch = hr_batch.to(device)
            if torch.cuda.is_available():
                lr_batch = lr_batch.to(device)
            image1 = netG(lr_batch[:, :3, :, :, :]).unsqueeze(1)
            image2 = netG(lr_batch[:, 1:4, :, :, :]).unsqueeze(1)
            image3 = netG(lr_batch[:, 2:5, :, :, :]).unsqueeze(1)

            fake_img = torch.cat((image1, image2, image3), dim=1)
            
            netD.zero_grad()
            # real_out = netD(hr_batch).mean()
            # fake_out = netD(fake_img).mean()
            real_out = netD(hr_batch)
            fake_out = netD(fake_img)
            print('real')
            print(real_out.mean())
            print('fake')
            print(fake_out.mean())

            real_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out))
            fake_loss = F.binary_cross_entropy(fake_out, torch.zeros_like(fake_out))

            d_loss = real_loss + fake_loss
            # d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)

            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runtime error in Google Colab ##
            image1 = netG(lr_batch[:, :3, :, :, :]).unsqueeze(1)
            image2 = netG(lr_batch[:, 1:4, :, :, :]).unsqueeze(1)
            image3 = netG(lr_batch[:, 2:5, :, :, :]).unsqueeze(1)

            fake_img = torch.cat((image1, image2, image3), dim=1)
            fake_out = netD(fake_img).mean()
            B, C, D, H, W = fake_img.shape
            new_shape = (B * D, C, H, W)
            ##
            g_loss = generator_criterion(fake_out, fake_img.reshape(new_shape), hr_batch.reshape(new_shape))
            g_loss.backward()
            optimizerG.step()
            print(f"Epoch {epoch}/{NUM_EPOCHS}, Step {step}/{total_steps}: Discriminator Loss = {d_loss.item():.4f}, Generator Loss = {g_loss.item():.4f}")

        epoch_duration = time.time() - epoch_start_time  # Calculate the duration of the epoch
        print(f"Epoch {epoch}/{NUM_EPOCHS} completed in {epoch_duration:.2f} seconds")  # Print the time per epoch
        torch.save(netG.state_dict(), 'new_dis4/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'new_dis4/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    