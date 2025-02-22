import os, sys, random
import math
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from PIL import Image, ImageEnhance

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths

def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

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

def img_resize(img, scale=2):
    """
    input: img, uint8, HxWx3(RGB)
    output: 
    """
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(h / scale), int(w / scale)), interpolation = cv2.INTER_LINEAR)
    return img

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
    # img = cv2.copyMakeBorder(img, 0, scale - h % scale, 0, scale - w % scale, cv2.BORDER_CONSTANT)

    return img

def img_save(img, img_path):
    """
    input: img, uint8, HxWx3(RGB)
    """
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def yuv444p_read(path, w, h):
    """
    input: path, weigth, height
    output: ndarray(uint8, HWC)
    """
    raw_bytes = np.fromfile(path, dtype="uint8")
    y = raw_bytes[0:w*h].reshape(h, w)
    u = raw_bytes[w*h:w*h*2].reshape(h, w)
    v = raw_bytes[w*h*2:w*h*3].reshape(h, w)
    yuv = np.stack([y, u, v], axis=2)
    return yuv

def yuv444p_save(yuv, path):
    """
    input: yuv(uint8, HWC), path
    """
    raw_bytes = yuv.transpose((2,0,1)).flatten().tobytes()
    with open(path, "wb") as f:
        f.write(raw_bytes)

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def single2tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

def rgb2yuv(rgb):
    '''
    Input:
        np.float32, [0, 1]
        shape [h, w, c], rgb
    '''
    assert rgb.dtype == np.float32
    m = np.array([[0.2989, -0.1688, 0.5000],
                 [0.5866, -0.3312, -0.4184],
                 [0.1145, 0.5000, -0.0816]], dtype=np.float32)
    yuv = np.dot(rgb, m)
    yuv[:, :, 0] = (yuv[:, :, 0] * 219. + 16.) / 255.
    yuv[:, :, 1] = (yuv[:, :, 1] * 224. + 128.) / 255.
    yuv[:, :, 2] = (yuv[:, :, 2] * 224. + 128.) / 255.
    return yuv

def yuv2rgb(yuv):
    '''
    Input:
        np.float32, [0, 1]
        shape [h, w, c], yuv
    '''
    assert yuv.dtype == np.float32
    yuv[:, :, 0] = (yuv[:, :, 0] * 255. - 16.) / 219.
    yuv[:, :, 1] = (yuv[:, :, 1] * 255. - 128.) / 224.
    yuv[:, :, 2] = (yuv[:, :, 2] * 255. - 128.) / 224.

    m = np.array([[1.0, 1.0, 1.0],
                 [0, -0.3456, 1.7710],
                 [1.4022, -0.7145, 0]], dtype=np.float32)
    rgb = np.dot(yuv, m)     
    return rgb

def rgb2yuv_torch(rgb):
    '''
    Input:
        torch.float32, [0, 1]
        shape [c, h, w], rgb
    '''
    assert rgb.dtype == torch.float32
    m = torch.tensor([[0.2989, -0.1688, 0.5000],
                 [0.5866, -0.3312, -0.4184],
                 [0.1145, 0.5000, -0.0816]], dtype=torch.float32)
    rgb = rgb.permute(1, 2, 0)
    yuv = torch.matmul(rgb, m)
    yuv[:, :, 0] = (yuv[:, :, 0] * 219. + 16.) / 255.
    yuv[:, :, 1] = (yuv[:, :, 1] * 224. + 128.) / 255.
    yuv[:, :, 2] = (yuv[:, :, 2] * 224. + 128.) / 255.
    yuv = yuv.permute(2, 0, 1)
    return yuv

def yuv2rgb_torch(yuv):
    '''
    Input:
        torch.float32, [0, 1]
        shape [c, h, w], yuv
    '''
    assert yuv.dtype == torch.float32
    yuv = yuv.permute(1, 2, 0)
    yuv[:, :, 0] = (yuv[:, :, 0] * 255. - 16.) / 219.
    yuv[:, :, 1] = (yuv[:, :, 1] * 255. - 128.) / 224.
    yuv[:, :, 2] = (yuv[:, :, 2] * 255. - 128.) / 224.

    m = torch.tensor([[1.0, 1.0, 1.0],
                 [0, -0.3456, 1.7710],
                 [1.4022, -0.7145, 0]], dtype=torch.float32)
    rgb = torch.matmul(yuv, m)    
    rgb = rgb.permute(2, 0, 1) 
    return rgb

def add_JPEG_noise(img, quality_factor):
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img

def unsharp_mask(image, radius=5, amount=1.0, threshold=0):
    """
    Input:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        radius (float): Kernel size of Gaussian blur.
        amount (float): Sharp weight.
    Return a sharpened version of the image, using an unsharp mask.
    """
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    if radius % 2 == 0:
        radius += 1
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 1)
    if threshold > 0:
        #np.copyto(sharpened, image, where=low_contrast_mask)
        low_contrast_mask = np.absolute(image - blurred) * 255 < threshold
        low_contrast_mask = low_contrast_mask.astype("float32")
        low_contrast_mask = cv2.GaussianBlur(low_contrast_mask, (radius, radius), 0)
        sharpened = (1 - low_contrast_mask) * sharpened +  low_contrast_mask * image
    return sharpened

def sharp_constrain(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(img,cv2.CV_64F,1,0))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(img,cv2.CV_64F,0,1))
    grad = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    weight_map = grad.astype(np.float32)/255.0
    weight_map = 1.0 - np.clip(weight_map, 0, 1.0)
    return cv2.cvtColor(weight_map, cv2.COLOR_GRAY2RGB).astype(np.float32)

def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    weight_map = sharp_constrain(img)
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    sharp_res = weight * residual
    constrained_res = np.multiply(sharp_res, weight_map)
    sharp = img + constrained_res
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img

def pil_sharpen(image, factor=1.5):
    """
    Input:
        Numpy array
    """
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(image)
    sharpened = enhancer.enhance(factor)
    sharpened = np.array(sharpened)
    return sharpened

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    mse += sys.float_info.epsilon
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return _ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(_ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]
    kernel = kernel.to(torch.float32)

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
