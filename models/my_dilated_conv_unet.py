import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.block(x)
        return x


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, padding):
        super(DilatedConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.block(x)
        return x




class ConcatDoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatDoubleConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(out_channels))

    def forward(self, x, skip):
        x = torch.cat((skip, x), dim=1)
        x = self.block(x)
        return x



class MyDilatedConvUNet(nn.Module):
    def __init__(self, filters=44, depth=3, bottleneck_depth=6):
        super(MyDilatedConvUNet, self).__init__()
        self.depth = depth
        self.encoder_path = nn.ModuleList()
        src_in_channels = 4     # Geo-TIFF has four channels (R, G, B, and NIR)
        for d in range(depth):
            in_channels = src_in_channels if d == 0 else filters * 2 ** (d-1)
            self.encoder_path.append(
                DoubleConvBlock(in_channels, filters * 2 ** d))
        self.maxpool = nn.MaxPool2d(2, 2, padding=0)
        self.bottleneck_path = nn.ModuleList()
        for d in range(bottleneck_depth):
            in_channels = filters * 2 ** (depth - 1) if d == 0 else filters * 2 ** depth
            self.bottleneck_path.append(DilatedConvBlock(in_channels, filters * 2 ** depth, 2 ** d, 2 ** d))
        self.decoder_path = nn.ModuleList()
        for d in range(depth):
            in_channels = filters * 2 ** (depth - d)
            self.decoder_path.append(ConcatDoubleConvBlock(in_channels, filters * 2 ** (depth - d - 1)))
        self.up_path = nn.ModuleList()
        for d in range(depth):
            in_channels = filters * 2 ** (depth - d)
            self.up_path.append(nn.ConvTranspose2d(in_channels, filters * 2 ** (depth - d - 1),
                                                        kernel_size=4, stride=2, padding=1))
        out_channels = 4     # output channels (num_classes + 1(background))
        self.last_conv = nn.Conv2d(filters, out_channels, kernel_size=1)

    def forward(self, x):
        skip = []
        for block in self.encoder_path:
            x = block(x)
            skip.append(x)
            x = self.maxpool(x)
        dilated = []
        for block in self.bottleneck_path:
            x = block(x)
            dilated.append(x)
        x = torch.stack(dilated, dim=-1).sum(dim=-1)  # sum over list

        # up-sampling and double convolutions
        for d in range(self.depth):
            x = self.up_path[d](x)
            x = self.decoder_path[d](x, skip[-(d+1)])

        return self.last_conv(x)