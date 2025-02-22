import os, sys, time
import argparse
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tvf
from torch import nn
import torch.nn.functional as F


###  python inference_pre.py --gpu 3 --input_lr_dir /data/docker/machengqian/dataset/FFDNet/testsets/CBSD68/ --output_sr_dir /data/docker/machengqian/preprocess-mmai/output/tmp

"""
Inference Utilities
"""

def img_read(path, n_channels=3):
    """
    input: path
    output: uint8, HxWx3(RGB or GGG), or HxWx1 (G)
    """
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

def img_save(img, img_path):
    """
    input: img, uint8, HxWx3(RGB)
    """
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def img_padding(img, scale=8):
    """
    input: img, uint8, HxWx3(RGB)
    output: 
    """
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(img, 0, scale - h % scale, 0, scale - w % scale, cv2.BORDER_CONSTANT)

    return img

def img_rm_padding(img, ori_img):
    """
    input: img, uint8, HxWx3(RGB)
    output: 
    """
    h, w = ori_img.shape[:2]
    img = img[:h, :w, :]
 
    return img

def uint2single(img):
    return np.float32(img/255.)

def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def load_checkpoint(path, model, device="cuda:0"):

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    return model

def inference(img, model, device):

    img_L = img_padding(img.copy())
    img_L = uint2single(img_L)
    img_L = tvf.to_tensor(img_L)
    img_L = img_L.to(device).unsqueeze(0)

    # print(img_L.shape)
    img_E = model(img_L)

    E = tensor2single(img_E)
    E = single2uint(E)

    E = img_rm_padding(E, img)

    return E


"""
Model Architecture
"""

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

class RFDB_lite_v2(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB_lite_v2, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*3, in_channels, 1)


    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        r_c3 = self.act(self.c4(r_c2))
        out = torch.cat([distilled_c1, distilled_c2, r_c3], dim=1)
        out_fused = self.c5(out)

        return out_fused

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


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


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
        self.fea_conv = pixel_unshuffle() 
        self.fea_conv_c = conv_block(in_nc * upscale**2, nf, kernel_size=1, act_type='lrelu')

        self.B1 = RFDB_lite_v2(in_channels=nf)
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_fea = self.fea_conv_c(out_fea)

        out_B1 = self.B1(out_fea)
        out_B = out_B1 + out_fea

        res = self.LR_conv(out_B)
        output = self.upsampler(res)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


def main():
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument("--gpu", default=0, type=int, help='gpu id')
    parser.add_argument("-i", "--input_lr_dir", type=str, default="./input", help="the folder of the input images")
    parser.add_argument("-o", "--output_sr_dir", type=str, default="./output", help="the folder of the output images")

    opt = parser.parse_args()
    gpu_id = opt.gpu
    device = torch.device("cuda", gpu_id)

    input_dir = opt.input_lr_dir
    output_dir = opt.output_sr_dir

    input_list = os.listdir(input_dir)
    
    model_folder = "./checkpoints/"
    model_name = "step_1600000"

    model_path = os.path.join(model_folder, model_name + '.pth')
    
    model = RFLite_v2()
    # model = torch.jit.load(model_path)
    model = load_checkpoint(model_path, model, device)
    model.eval()
    model = model.to(device)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    for img_path in input_list:
        save_path = os.path.join(output_dir, img_path)
        with torch.no_grad():
            
            img = img_read(os.path.join(input_dir, img_path))
            start = time.time()
            img_E = inference(img, model, device)
            end = time.time()
            print("inference duration: ", end - start)
            img_save(img_E, save_path)
    

if __name__ == "__main__":
    main()