# -*- coding: utf-8 -*-
from utils import utils_image
from utils import utils_deblur
import random
import math
import numpy as np
import torch
import cv2
import scipy
import scipy.stats as ss
import scipy.io as io
from scipy import ndimage
from scipy.interpolate import interp2d
from utils.degradations import circular_lowpass_kernel, random_mixed_kernels

"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


"""
# --------------------------------------------
# calculate PCA projection matrix
# --------------------------------------------
"""


def get_pca_matrix(x, dim_pca=15):
    """
    Args:
        x: 225x10000 matrix
        dim_pca: 15
    Returns:
        pca_matrix: 15x225
    """
    C = np.dot(x, x.T)
    w, v = scipy.linalg.eigh(C)
    pca_matrix = v[:, -dim_pca:].T

    return pca_matrix


def show_pca(x):
    """
    x: PCA projection matrix, e.g., 15x225
    """
    for i in range(x.shape[0]):
        xc = np.reshape(x[i, :], (int(np.sqrt(x.shape[1])), -1), order="F")
        utils_image.surf(xc)


def cal_pca_matrix(path='PCA_matrix.mat', ksize=15, l_max=12.0, dim_pca=15, num_samples=500):
    kernels = np.zeros([ksize*ksize, num_samples], dtype=np.float32)
    for i in range(num_samples):

        theta = np.pi*np.random.rand(1)
        l1    = 0.1+l_max*np.random.rand(1)
        l2    = 0.1+(l1-0.1)*np.random.rand(1)

        k = anisotropic_Gaussian(ksize=ksize, theta=theta[0], l1=l1[0], l2=l2[0])

        # util.imshow(k)

        kernels[:, i] = np.reshape(k, (-1), order="F")  # k.flatten(order='F')

    # io.savemat('k.mat', {'k': kernels})

    pca_matrix = get_pca_matrix(kernels, dim_pca=dim_pca)

    io.savemat(path, {'p': pca_matrix})

    return pca_matrix


"""
# --------------------------------------------
# shifted anisotropic Gaussian kernels
# --------------------------------------------
"""


def shifted_anisotropic_Gaussian(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def gen_kernel(k_size=np.array([25, 25]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=12., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    sf = random.choice([1, 2, 3, 4])
    scale_factor = np.array([sf, sf])
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = 0#-noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""

def anime_train_degradation(img_H, sf=2, gaussion_noise_max=0.08, enhance_hr=False):
    """
    input: img_H, float[0, 1], HxWxC
    output: img_H, img_L
    """
    img_L = img_H.copy()

    # enhance HR image
    if enhance_hr:
        img_H = utils_image.unsharp_mask(img_H, amount=0.1, threshold=0)

    # downsampling, 5/8 uses blur+nearest, 3/8 uses interpolation only
    if random.randint(0, 7) > 2:
        # generate kernel
        sf_k = random.choice([2, 3, 4])
        k = gen_kernel(scale_factor=np.array([sf_k, sf_k]))  # gaussian blur
        mode_k = random.randint(0, 7)
        k = utils_image.augment_img(k, mode=mode_k)
        k = np.expand_dims(k, axis=2)
        k = k.astype(np.float32)

        # blur
        img_L = ndimage.filters.convolve(img_L, k, mode='wrap')

        # nearest downsampling
        img_L = img_L[0::sf, 0::sf, ...]
    else:
        # interpolation downsampling
        assert img_L.shape[0] % sf == 0 and img_L.shape[1] % sf == 0
        img_L = cv2.resize(img_L, (img_L.shape[1] // sf, img_L.shape[0] // sf), 
            interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))

    # noise level
    if random.randint(0, 7) > 6:
        noise_level = 0.0
    else:
        noise_level = random.uniform(1, 255) / 255.0
    
    if noise_level > 0.0:
        if random.randint(0, 7) > 4:
            # add Gaussian noise
            img_L = img_L + np.random.normal(0, noise_level * gaussion_noise_max, img_L.shape).astype(np.float32)
            img_L = np.clip(img_L, 0.0, 1.0)
        else:
            # add JPEG noise
            # quality_factor = int((1.0 - noise_level) * (95 - 30) + 30)  # quality 30-95
            quality_factor = random.randint(15, 95)
            img_L = utils_image.add_JPEG_noise(img_L, quality_factor)

    return img_H, img_L


def anime_val_degradation(img_H, sf=2, gaussion_noise_max=0.08, enhance_hr=False):
    img_L = img_H.copy()

    # enhance HR image
    if enhance_hr:
        img_H = utils_image.unsharp_mask(img_H, amount=0.5, threshold=0)

    # bicubic downsampling
    assert img_L.shape[0] % sf == 0 and img_L.shape[1] % sf == 0
    img_L = cv2.resize(img_L, (img_L.shape[1] // sf, img_L.shape[0] // sf), interpolation=cv2.INTER_CUBIC)
    # print("L: ", img_L.shape)
    # print("H: ", img_H.shape)
    # add JPEG noise
    # noise_level = 128.0 / 255.0
    # quality_factor = int((1.0 - noise_level) * (95 - 30) + 30)  # quality 30-95
    quality_factor = random.randint(0, 7)
    img_L = utils_image.add_JPEG_noise(img_L, quality_factor)

    return img_H, img_L


def real_degradation(opt, sf=1):

    blur_kernel_size = opt['blur_kernel_size']
    kernel_list = opt['kernel_list']
    kernel_prob = opt['kernel_prob']  # a list for each kernel probability
    blur_sigma = opt['blur_sigma']
    betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
    betap_range = opt['betap_range']  # betap used in plateau blur kernels
    sinc_prob = opt['sinc_prob']

     # blur settings for the second degradation
    blur_kernel_size2 = opt['blur_kernel_size2']
    kernel_list2 = opt['kernel_list2']
    kernel_prob2 = opt['kernel_prob2']
    blur_sigma2 = opt['blur_sigma2']
    betag_range2 = opt['betag_range2']
    betap_range2 = opt['betap_range2']
    sinc_prob2 = opt['sinc_prob2']

    # a final sinc filter
    final_sinc_prob = opt['final_sinc_prob']

    pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1
    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    
    # sharpen HR image
    # if enhance_hr:
    #     # img_H = utils_image.unsharp_mask(img_H, amount=0.5, threshold=0)
    #     img_H = utils_image.usm_sharp(img_H, weight=0.4, threshold=15)
    
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma, [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))


    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list2,
            kernel_prob2,
            kernel_size,
            blur_sigma2,
            blur_sigma2, [-math.pi, math.pi],
            betag_range2,
            betap_range2,
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))


    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor


    return kernel, kernel2, sinc_kernel
