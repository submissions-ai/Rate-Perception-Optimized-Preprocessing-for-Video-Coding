from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


# @ARCH_REGISTRY.register()
class UNetSN(nn.Module):
    """Defines a U-Net with spectral normalization (SN)

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        # self.conv8 = norm(nn.Conv2d(num_feat, num_in_ch, 3, 1, 1, bias=False))
        self.conv8 = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1, bias=False)
        # self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        # print("conv0 shape: ", x0.shape)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        # print("conv1 shape: ", x1.shape)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        # print("conv2 shape: ", x2.shape)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        # print("conv3 shape: ", x3.shape)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        # print("conv4 shape: ", x4.shape)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        # print("conv5 shape: ", x5.shape)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        # print("conv6 shape: ", x6.shape)
        if self.skip_connection:
            x6 = x6 + x0
        
        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        # print("conv7 shape: ", out.shape)
        out = self.conv8(out)
        # out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        # out = self.conv9(out)

        return out

class UNetPS(nn.Module):
    """Defines a U-Net with Pixel Shuffle upsample

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetPS, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)
        # upsample
        self.conv4 = nn.Conv2d(num_feat * 8, num_feat * 16, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        # extra convolutions
        self.conv7 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        # self.conv8 = norm(nn.Conv2d(num_feat, num_in_ch, 3, 1, 1, bias=False))
        self.conv8 = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1, bias=False)
        # pixel shuffle
        self.pixel_shuffle = nn.PixelShuffle(2)
        # self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        # print("conv0 shape: ", x0.shape)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        # print("conv1 shape: ", x1.shape)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        # print("conv2 shape: ", x2.shape)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        # print("conv3 shape: ", x3.shape)

        # upsample
        x4_pre = self.conv4(x3)
        x4 = F.leaky_relu(self.pixel_shuffle(x4_pre), negative_slope=0.2, inplace=True)
        # print("conv4 shape: ", x4.shape)
        if self.skip_connection:
            x4 = x4 + x2
        x5_pre = self.conv5(x4)
        x5 = F.leaky_relu(self.pixel_shuffle(x5_pre), negative_slope=0.2, inplace=True)
        # print("conv5 shape: ", x5.shape)
        if self.skip_connection:
            x5 = x5 + x1
        x6_pre = self.conv6(x5)
        x6 = F.leaky_relu(self.pixel_shuffle(x6_pre), negative_slope=0.2, inplace=True)
        # print("conv6 shape: ", x6.shape)
        if self.skip_connection:
            x6 = x6 + x0
        
        # extra convolutions
        x7 = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        # print("conv7 shape: ", out.shape)
        out = self.conv8(x7)
        # out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        # out = self.conv9(out)

        return out


class UNet(nn.Module):
    """Defines a U-Net with Pixel Shuffle upsample

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNet, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)
        # upsample
        self.conv4 = nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False)
        # extra convolutions
        self.conv7 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        # self.conv8 = norm(nn.Conv2d(num_feat, num_in_ch, 3, 1, 1, bias=False))
        self.conv8 = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1, bias=False)
        # self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        # print("conv0 shape: ", x0.shape)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        # print("conv1 shape: ", x1.shape)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        # print("conv2 shape: ", x2.shape)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        # print("conv3 shape: ", x3.shape)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        # print("conv4 shape: ", x4.shape)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        # print("conv5 shape: ", x5.shape)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        # print("conv6 shape: ", x6.shape)
        if self.skip_connection:
            x6 = x6 + x0
        
        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        # print("conv7 shape: ", out.shape)
        out = self.conv8(out)
        # out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        # out = self.conv9(out)

        return out

