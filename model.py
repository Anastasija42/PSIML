import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.up = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.up(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1
    ):
        super(UNET, self).__init__()

        self.inc = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.semifinal = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        def same_size(a, b):
            if a.shape != b.shape:
                F.resize(a, size=b.shape[2:])

        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)
        x = self.down1(x)
        skip_connections.append(x)
        x = self.down2(x)
        skip_connections.append(x)
        x = self.down3(x)
        skip_connections.append(x)
        x = self.down4(x)

        skip_connections = skip_connections[::-1]

        x = self.up0(x)
        skip_connection = skip_connections[0]
        same_size(x, skip_connection)
        x = torch.cat((skip_connection, x), dim=1)

        x = self.up1(x)
        skip_connection = skip_connections[1]
        same_size(x, skip_connection)
        x = torch.cat((skip_connection, x), dim=1)

        x = self.up2(x)
        skip_connection = skip_connections[2]
        same_size(x, skip_connection)
        x = torch.cat((skip_connection, x), dim=1)

        x = self.up3(x)
        skip_connection = skip_connections[3]
        same_size(x, skip_connection)
        x = torch.cat((skip_connection, x), dim=1)

        del skip_connections
        x = self.semifinal(x)
        x = self.final(x)

        return x
