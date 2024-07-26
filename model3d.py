import math
import torch
from torch import nn


class Generator3D(nn.Module):
    def __init__(self, scale_factor):
        super(Generator3D, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        
        self.frame_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.frame_conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.frame_conv3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.combine_conv = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        frame1_output = self.frame_conv1(x[:, 0, :, :, :])
        frame2_output = self.frame_conv2(x[:, 1, :, :, :])
        frame3_output = self.frame_conv3(x[:, 2, :, :, :])
        
        stacked_output = torch.cat((frame1_output, frame2_output, frame3_output), dim=1)
        
        combined_output = self.combine_conv(stacked_output)
        
        block2 = self.block2(combined_output)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7 + combined_output)

        return (torch.tanh(block8) + 1) / 2


class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()
        self.frame_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2)
        )
        self.frame_conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2)
        )
        self.frame_conv3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2)
        )
        
        self.combine_conv = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.net = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)

            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(256, 256, kernel_size=1),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        frame1_output = self.frame_conv1(x[:, 0, :, :, :])
        frame2_output = self.frame_conv2(x[:, 1, :, :, :])
        frame3_output = self.frame_conv3(x[:, 2, :, :, :])
        stacked_output = torch.cat((frame1_output, frame2_output, frame3_output), dim=1)
        
        combined_output = self.combine_conv(stacked_output)
        return torch.sigmoid(self.net(combined_output).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
