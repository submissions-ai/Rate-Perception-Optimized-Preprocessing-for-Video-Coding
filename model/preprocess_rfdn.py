import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F


class RGB2YUV(nn.Module):
    def __init__(self):
        super(RGB2YUV, self).__init__()
    
    def forward(self, x):
        '''
        Input:
            torch.float32, [0, 1]
            shape [c, h, w], rgb
        '''
        assert x.dtype == torch.float32
        m = torch.tensor([[0.2989, -0.1688, 0.5000],
                    [0.5866, -0.3312, -0.4184],
                    [0.1145, 0.5000, -0.0816]], dtype=torch.float32)
        # rgb = x.permute(1, 2, 0)
        rgb = x.permute(0, 2, 3, 1)
        yuv = torch.matmul(rgb, m)
        yuv[:, :, :, 0] = (yuv[:, :, :, 0] * 219. + 16.) / 255.
        yuv[:, :, :, 1] = (yuv[:, :, :, 1] * 224. + 128.) / 255.
        yuv[:, :, :, 2] = (yuv[:, :, :, 2] * 224. + 128.) / 255.
        x = yuv.permute(0, 3, 1, 2)
        # rgb = x.permute(1, 2, 0)
        # yuv = torch.matmul(rgb, m)
        # yuv[:, :, 0] = (yuv[:, :, 0] * 219. + 16.) / 255.
        # yuv[:, :, 1] = (yuv[:, :, 1] * 224. + 128.) / 255.
        # yuv[:, :, 2] = (yuv[:, :, 2] * 224. + 128.) / 255.
        # x = yuv.permute(2, 0, 1)
        return x

class YUV2RGB(nn.Module):
    def __init__(self):
        super(YUV2RGB, self).__init__()
    
    def forward(self, x):
        '''
        Input:
            torch.float32, [0, 1]
            shape [c, h, w], yuv
        '''
        assert x.dtype == torch.float32
        yuv = x.permute(0, 2, 3, 1)
        yuv[:, :, :, 0] = (yuv[:, :, :, 0] * 255. - 16.) / 219.
        yuv[:, :, :, 1] = (yuv[:, :, :, 1] * 255. - 128.) / 224.
        yuv[:, :, :, 2] = (yuv[:, :, :, 2] * 255. - 128.) / 224.
        m = torch.tensor([[1.0, 1.0, 1.0],
                    [0, -0.3456, 1.7710],
                    [1.4022, -0.7145, 0]], dtype=torch.float32)
        rgb = torch.matmul(yuv, m)    
        x = rgb.permute(0, 3, 1, 2)
        # yuv = x.permute(1, 2, 0)
        # yuv[:, :, 0] = (yuv[:, :, 0] * 255. - 16.) / 219.
        # yuv[:, :, 1] = (yuv[:, :, 1] * 255. - 128.) / 224.
        # yuv[:, :, 2] = (yuv[:, :, 2] * 255. - 128.) / 224.

        # m = torch.tensor([[1.0, 1.0, 1.0],
        #             [0, -0.3456, 1.7710],
        #             [1.4022, -0.7145, 0]], dtype=torch.float32)
        # rgb = torch.matmul(yuv, m)    
        # x = rgb.permute(2, 0, 1) 
        return x

# def rgb2yuv_torch(rgb):
#     '''
#     Input:
#         torch.float32, [0, 1]
#         shape [c, h, w], rgb
#     '''
#     assert rgb.dtype == torch.float32
#     m = torch.tensor([[0.2989, -0.1688, 0.5000],
#                  [0.5866, -0.3312, -0.4184],
#                  [0.1145, 0.5000, -0.0816]], dtype=torch.float32)
#     rgb = rgb.permute(1, 2, 0)
#     yuv = torch.matmul(rgb, m)
#     yuv[:, :, 0] = (yuv[:, :, 0] * 219. + 16.) / 255.
#     yuv[:, :, 1] = (yuv[:, :, 1] * 224. + 128.) / 255.
#     yuv[:, :, 2] = (yuv[:, :, 2] * 224. + 128.) / 255.
#     yuv = yuv.permute(2, 0, 1)
#     return yuv

# def yuv2rgb_torch(yuv):
#     '''
#     Input:
#         torch.float32, [0, 1]
#         shape [c, h, w], yuv
#     '''
#     assert yuv.dtype == torch.float32
#     yuv = yuv.permute(1, 2, 0)
#     yuv[:, :, 0] = (yuv[:, :, 0] * 255. - 16.) / 219.
#     yuv[:, :, 1] = (yuv[:, :, 1] * 255. - 128.) / 224.
#     yuv[:, :, 2] = (yuv[:, :, 2] * 255. - 128.) / 224.

#     m = torch.tensor([[1.0, 1.0, 1.0],
#                  [0, -0.3456, 1.7710],
#                  [1.4022, -0.7145, 0]], dtype=torch.float32)
#     rgb = torch.matmul(yuv, m)    
#     rgb = rgb.permute(2, 0, 1) 
#     return rgb


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m


class RFDB_lite(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB_lite, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        # self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        # self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*3, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        # distilled_c3 = self.act(self.c3_d(r_c2))
        # r_c3 = (self.c3_r(r_c2))
        # r_c3 = self.act(r_c3+r_c2)

        # r_c4 = self.act(self.c4(r_c3))

        # out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        # out_fused = self.esa(self.c5(out)) 
        r_c3 = self.act(self.c4(r_c2))
        out = torch.cat([distilled_c1, distilled_c2, r_c3], dim=1)
        out_fused = self.c5(out)

        return out_fused


class RFDB_lite_v2(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB_lite_v2, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        # self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        # self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*3, in_channels, 1)
        # self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        # distilled_c3 = self.act(self.c3_d(r_c2))
        # r_c3 = (self.c3_r(r_c2))
        # r_c3 = self.act(r_c3+r_c2)

        # r_c4 = self.act(self.c4(r_c3))

        # out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        # out_fused = self.esa(self.c5(out)) 
        r_c3 = self.act(self.c4(r_c2))
        out = torch.cat([distilled_c1, distilled_c2, r_c3], dim=1)
        out_fused = self.c5(out)

        return out_fused


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused

# def pixel_unshuffle(input, downscale_factor):
#     '''
#     input: batchSize * c * k*w * k*h
#     kdownscale_factor: k
#     batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
#     '''
#     c = input.shape[1]

#     kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
#                                1, downscale_factor, downscale_factor],
#                          device=input.device)
#     for y in range(downscale_factor):
#         for x in range(downscale_factor):
#             kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
#     return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return nn.PixelUnshuffle(self.downscale_factor)
        # return pixel_unshuffle(input, self.downscale_factor)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

# def pixelunshuffle_block(in_channels, out_channels, downscale_factor=2, kernel_size=1, stride=1):
#     # pixel_unshuffle = PixelUnshuffle(downscale_factor)
#     pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
#     conv = conv_layer(in_channels*(downscale_factor**2), out_channels, kernel_size, stride)
#     return sequential(pixel_unshuffle, conv)

class pixel_unshuffle(nn.Module):
    def __init__(self, scale=2):
        super(pixel_unshuffle, self).__init__()
        self.scale = scale
                                
    def forward(self, x):
        n, c, h, w = x.shape
        x = torch.reshape(x, (n, c, h // self.scale, self.scale, w // self.scale, self.scale))
        x = x.permute((0, 1, 3, 5, 2, 4))
        x = torch.reshape(x, (n, c * self.scale * self.scale, h // self.scale, w // self.scale))

        return x


class RFLite_v2(nn.Module):
    def __init__(self, in_nc=3, nf=8, num_modules=2, out_nc=3, upscale=2):
        super(RFLite_v2, self).__init__()

        # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3, stride=2)
        # self.fea_conv = B.pixelunshuffle_block(in_nc, nf, 2, kernel_size=3)
        # self.fea_conv = pixelunshuffle_block(in_nc, nf, 2, kernel_size=1)
        self.fea_conv = pixel_unshuffle() 
        self.fea_conv_c = conv_block(in_nc * upscale**2, nf, kernel_size=1, act_type='lrelu')

        self.B1 = RFDB_lite_v2(in_channels=nf)
        # self.B2 = RFDB_lite_v2(in_channels=nf)
        # self.B3 = RFDB(in_channels=nf)
        # self.B4 = B.RFDB(in_channels=nf)
        # self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        # upsample_block = pixelshuffle_block
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_fea = self.fea_conv_c(out_fea)
        # print(out_fea.shape)
        out_B1 = self.B1(out_fea)
        # out_B2 = self.B2(out_B1)
        # out_B3 = self.B3(out_B2)
        # out_B4 = self.B4(out_B3)

        # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        # out_B = self.c(torch.cat([out_B1, out_B2], dim=1))
        # out_B1 = self.c(torch.cat([out_B1], dim=1))
        out_B = out_B1 + out_fea
        # out_lr = self.LR_conv(out_B) + out_fea
        # output = self.upsampler(out_lr)

        res = self.LR_conv(out_B)
        output = self.upsampler(res)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


# class RFLite_v3(nn.Module):
#     def __init__(self, in_nc=3, nf=8, num_modules=2, out_nc=3, upscale=2):
#         super(RFLite_v3, self).__init__()

#         self.fea_conv = conv_layer(in_nc, nf, kernel_size=3, stride=2)
#         # self.fea_conv = B.pixelunshuffle_block(in_nc, nf, 2, kernel_size=3)
#         # self.fea_conv = pixelunshuffle_block(in_nc, nf, 2, kernel_size=1)
#         # self.fea_conv = pixel_unshuffle() 
#         # self.fea_conv_c = conv_block(in_nc * upscale**2, nf, kernel_size=1, act_type='lrelu')

#         self.B1 = RFDB_lite_v2(in_channels=nf)
#         # self.B2 = RFDB_lite_v2(in_channels=nf)
#         # self.B3 = RFDB(in_channels=nf)
#         # self.B4 = B.RFDB(in_channels=nf)
#         # self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

#         self.LR_conv = conv_layer(nf, nf, kernel_size=3)

#         # upsample_block = pixelshuffle_block
#         self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
#         self.scale_idx = 0


#     def forward(self, input):
#         out_fea = self.fea_conv(input)
#         # out_fea = self.fea_conv_c(out_fea)
#         # print(out_fea.shape)
#         out_B1 = self.B1(out_fea)
#         # out_B2 = self.B2(out_B1)
#         # out_B3 = self.B3(out_B2)
#         # out_B4 = self.B4(out_B3)

#         # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
#         # out_B = self.c(torch.cat([out_B1, out_B2], dim=1))
#         # out_B1 = self.c(torch.cat([out_B1], dim=1))
#         out_B = out_B1 + out_fea
#         # out_lr = self.LR_conv(out_B) + out_fea
#         # output = self.upsampler(out_lr)

#         res = self.LR_conv(out_B)
#         output = self.upsampler(res)
#         return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

# class RFLite_v4(nn.Module):
#     def __init__(self, in_nc=3, nf=8, num_modules=2, out_nc=3, upscale=2):
#         super(RFLite_v4, self).__init__()
#         # self.yuv_to_rgb = YUV2RGB()
#         # self.rgb_to_yuv = RGB2YUV()
#         # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3, stride=2)
#         # self.fea_conv = B.pixelunshuffle_block(in_nc, nf, 2, kernel_size=3)
#         # self.fea_conv = pixelunshuffle_block(in_nc, nf, 2, kernel_size=1)
#         self.fea_conv = pixel_unshuffle() 
#         self.fea_conv_c = conv_block(in_nc * upscale**2, nf, kernel_size=1, act_type='lrelu')

#         self.B1 = RFDB_lite_v2(in_channels=nf)
#         # self.B2 = RFDB_lite_v2(in_channels=nf)
#         # self.B3 = RFDB(in_channels=nf)
#         # self.B4 = B.RFDB(in_channels=nf)
#         # self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

#         self.LR_conv = conv_layer(nf, nf, kernel_size=3)

#         # upsample_block = pixelshuffle_block
#         self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
#         self.scale_idx = 0


#     def forward(self, input):
#         # input = self.yuv_to_rgb(input)

#         out_fea = self.fea_conv(input)
#         out_fea = self.fea_conv_c(out_fea)
#         # print(out_fea.shape)
#         out_B1 = self.B1(out_fea)
#         # out_B2 = self.B2(out_B1)
#         # out_B3 = self.B3(out_B2)
#         # out_B4 = self.B4(out_B3)

#         # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
#         # out_B = self.c(torch.cat([out_B1, out_B2], dim=1))
#         # out_B1 = self.c(torch.cat([out_B1], dim=1))
#         out_B = out_B1 + out_fea
#         # out_lr = self.LR_conv(out_B) + out_fea
#         # output = self.upsampler(out_lr)

#         res = self.LR_conv(out_B)
#         output = self.upsampler(res)
#         # output = output.permute(0, 2, 3, 1)
#         # output = self.rgb_to_yuv(output)
#         return output

#     def set_scale(self, scale_idx):
#         self.scale_idx = scale_idx

class RFLite_v5(nn.Module):
    def __init__(self, in_nc=3, nf=8, num_modules=2, out_nc=3, upscale=2):
        super(RFLite_v5, self).__init__()

        # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3, stride=2)
        # self.fea_conv = B.pixelunshuffle_block(in_nc, nf, 2, kernel_size=3)
        # self.fea_conv = pixelunshuffle_block(in_nc, nf, 2, kernel_size=1)
        self.fea_conv = pixel_unshuffle() 
        self.fea_conv_c = conv_block(in_nc * upscale**2, nf, kernel_size=1, act_type='lrelu')

        self.B1 = RFDB_lite_v2(in_channels=nf)
        # self.B2 = RFDB_lite_v2(in_channels=nf)
        # self.B3 = RFDB(in_channels=nf)
        # self.B4 = B.RFDB(in_channels=nf)
        # self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        # upsample_block = pixelshuffle_block
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        input_norm = input / 255
        input_norm = input_norm.permute(0, 3, 1, 2)
        out_fea = self.fea_conv(input_norm)
        out_fea = self.fea_conv_c(out_fea)
        # print(out_fea.shape)
        out_B1 = self.B1(out_fea)
        # out_B2 = self.B2(out_B1)
        # out_B3 = self.B3(out_B2)
        # out_B4 = self.B4(out_B3)

        # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        # out_B = self.c(torch.cat([out_B1, out_B2], dim=1))
        # out_B1 = self.c(torch.cat([out_B1], dim=1))
        out_B = out_B1 + out_fea
        # out_lr = self.LR_conv(out_B) + out_fea
        # output = self.upsampler(out_lr)

        res = self.LR_conv(out_B)
        output = self.upsampler(res)
        output = torch.clip(output * 255, 0, 255)
        output = output.permute(0, 2, 3, 1)
        output = output.to(torch.uint8)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

class RFLite_v3(nn.Module):
    def __init__(self, in_nc=3, nf=16, num_modules=3, out_nc=3, upscale=2):
        super(RFLite_v3, self).__init__()

        # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3, stride=2)
        # self.fea_conv = B.pixelunshuffle_block(in_nc, nf, 2, kernel_size=3)
        # self.fea_conv = pixelunshuffle_block(in_nc, nf, 2, kernel_size=1)
        self.fea_unshuffle = pixel_unshuffle() 
        self.fea_conv_c = conv_block(in_nc * upscale**2, nf, kernel_size=1, act_type='lrelu')

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        # self.B4 = B.RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        # upsample_block = pixelshuffle_block
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_unshuffle(input)
        out_fea = self.fea_conv_c(out_fea)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        # out_B4 = self.B4(out_B3)

        # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3], dim=1))
        # out_lr = self.LR_conv(out_B) + out_fea
        # output = self.upsampler(out_lr)
        out_B_res = out_B + out_fea 

        out_res = self.LR_conv(out_B_res)
        output = self.upsampler(out_res)
        # output = up_res + input
        
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

# class RFLite_v5(nn.Module):
#     def __init__(self, in_nc=3, nf=8, num_modules=2, out_nc=3, upscale=2):
#         super(RFLite_v5, self).__init__()

#         self.fea_conv = pixel_unshuffle() 
#         self.fea_conv_c = conv_block(in_nc * upscale**2, nf, kernel_size=1, act_type='lrelu')
#         self.B1 = RFDB_lite_v2(in_channels=nf)
#         self.LR_conv = conv_layer(nf, nf, kernel_size=3)

#         self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
#         self.scale_idx = 0
#         # self.contrast_param = 1.015
#         # self.contrast = self.contrast_param * 256 * 16
#         # self.brightness = (100 * 511 / 200 - 128 - contrast / 32) / 255


#     def forward(self, input):
#         # input = input / 255
#         out_fea = self.fea_conv(input)
#         out_fea = self.fea_conv_c(out_fea)

#         out_B1 = self.B1(out_fea)
#         out_B = out_B1 + out_fea

#         res = self.LR_conv(out_B)
#         output = self.upsampler(res)
#         input_quant = (input * 255).int()
#         output_quant = (output * 255).int()
#         # output_diff = ((output_quant - input_quant) / 2).int() + output_quant
#         output_diff = (input_quant / 2).int() + (output_quant / 2).int()
#         output_diff = output_diff.float() / 255
#         # output_diff = input / 2 + output / 2
#         # output = torch.clip(output * 255, 0, 1)
#         ### contrast
#         # output = output * self.contrast_param + self.brightness
#         return output_diff

#     def set_scale(self, scale_idx):
#         self.scale_idx = scale_idx


class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=2):
        super(RFDN, self).__init__()

        # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3, stride=2)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output


class toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False, padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=False, padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential( 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, bias=False, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential( 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=False, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential( 
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, bias=False, padding=(1,1)),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        c1o = self.conv1(x)
        c2o = self.conv2(c1o)
        c3o = self.conv3(c2o)
        c4o = self.conv4(c3o)
        c5o = self.conv5(c4o)
        
        return c5o
        

if __name__=="__main__":

    # model = toy()
    model = RFLite()
    model = model.eval()
    src = torch.randn(1, 3, 1080, 1920).to("cpu", torch.float32)
    
    out = model(src)
    
    # for name, param in model.named_parameters():
    #     print(name)
    # print(param)
    torch.onnx.export(model, src, "./openvino_rflite.onnx", opset_version=12, export_params=True, input_names=['input'], output_names=["output"])
