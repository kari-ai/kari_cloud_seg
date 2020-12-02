import torch
import torch.nn as nn
from torchvision import models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class MyTernausNet(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super(MyTernausNet, self).__init__()
        self.encoder = models.vgg11(pretrained=pretrained).features

        self.conv1 = self.encoder[0]
        self.relu = self.encoder[1]
        self.conv2 = self.encoder[3]
        self.conv3_1 = self.encoder[6]
        self.conv3_2 = self.encoder[8]
        self.conv4_1 = self.encoder[11]
        self.conv4_2 = self.encoder[13]
        self.conv5_1 = self.encoder[16]
        self.conv5_2 = self.encoder[18]

        self.maxpool = nn.MaxPool2d(2, 2)

        self.center = DecoderBlock(512, 512, 256)
        self.decoder5 = DecoderBlock(256 + 512, 512, 256)
        self.decoder4 = DecoderBlock(256 + 512, 512, 128)
        self.decoder3 = DecoderBlock(128 + 256, 256, 64)
        self.decoder2 = DecoderBlock(64 + 128, 128, 32)
        self.decoder1 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, 1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, 1, 1, 1, 1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        skip1 = out
        out = self.maxpool(out)
        out = self.relu(self.conv2(out))
        skip2 = out
        out = self.maxpool(out)
        out = self.relu(self.conv3_1(out))
        out = self.relu(self.conv3_2(out))
        skip3 = out
        out = self.maxpool(out)
        out = self.relu(self.conv4_1(out))
        out = self.relu(self.conv4_2(out))
        skip4 = out
        out = self.maxpool(out)
        out = self.relu(self.conv5_1(out))
        out = self.relu(self.conv5_2(out))
        skip5 = out
        out = self.maxpool(out)

        # center
        out = self.center(out)

        # decoder
        out = self.decoder5(torch.cat([out, skip5], 1))
        out = self.decoder4(torch.cat([out, skip4], 1))
        out = self.decoder3(torch.cat([out, skip3], 1))
        out = self.decoder2(torch.cat([out, skip2], 1))
        out = self.decoder1(torch.cat([out, skip1], 1))

        return self.final(out)
