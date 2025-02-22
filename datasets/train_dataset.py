import os, sys, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy import ndimage
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from utils import utils_image, utils_deblur, utils_sisr

TRAIN_DATA_LIST_FILENAME = "flicker_div_data_patch_list.txt"

class TrainDataset_real(Dataset):

    def __init__(self, config, opt):
        super().__init__()

        self.opt = opt
        # self.data_list_path = os.path.join(config["dataset_root"], TRAIN_DATA_LIST_FILENAME)
        self.data_list_path = config["dataset_list_path"]
        self.data_list = self._load_data_list(self.data_list_path)
        # self.data_list = sorted(os.listdir(config["dataset_root"]))
        self.batch_size = config["batch_size"]
        self.patch_size = config["patch_size"]
        self.enhance_hr = config["enhance_hr"]
        self.paired_data = config["paired_data"]
        self.to_yuv = config["to_yuv"]
        self.sf = config["network"]["scale"]
        # kernel size
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.gaussion_noise_max = 50. / 255.

    def _load_data_list(self, data_list_path):
        with open(data_list_path, "r") as f:
            lines = f.readlines()

        data_list = []
        for line in lines:
            data_item = line.strip("\n ").split(",")
            data_list.append(data_item)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        h_path, l_path = self.data_list[index]
        # h_path = self.data_list[index]
        img_H = utils_image.img_read(h_path)

        # augmentation - flip, rotate
        mode = random.randint(0, 7)
        img_H = utils_image.augment_img(img_H, mode=mode)
        
        # uint8 to float32
        img_H = utils_image.uint2single(img_H)

        # randomly crop a patch
        h, w, _ = img_H.shape
        if self.patch_size % self.sf != 0:
            raise ValueError("patch size is unmatched!")
        l_patch_y = random.randint(0, max(0, h - self.patch_size//self.sf))
        l_patch_x = random.randint(0, max(0, w - self.patch_size//self.sf))
        h_patch_y = l_patch_y * self.sf
        h_patch_x = l_patch_x * self.sf

        patch_H = img_H[h_patch_y:h_patch_y + self.patch_size, h_patch_x:h_patch_x + self.patch_size, :]
        patch_L = patch_H.copy()
        # patch_L = img_L[l_patch_y:l_patch_y + self.patch_size//self.sf, l_patch_x:l_patch_x + self.patch_size//self.sf, :]

        # degradation
        # img_H, img_L = utils_sisr.anime_train_degradation(img_H, self.sf, self.gaussion_noise_max, self.enhance_hr)
        patch_H, kernel, kernel2, sinc_kernel = utils_sisr.real_degradation(patch_H, self.opt, self.sf, self.enhance_hr)
        

        # rgb to yuv
        if self.to_yuv:
            patch_H = utils_image.rgb2yuv(patch_H)
            # patch_L = utils_image.rgb2yuv(patch_L)

        # convert to tensor
        patch_H = utils_image.single2tensor(patch_H)
        patch_L = utils_image.single2tensor(patch_L)

        item = {
            "H": patch_H,
            "L": patch_L,
            "kernel": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel
        }

        return item


class TrainDataset_real_paired(Dataset):

    def __init__(self, config, opt):
        super().__init__()

        self.opt = opt
        self.data_list_path = config["dataset_list_path"]
        self.data_list = self._load_data_list(self.data_list_path)
        # self.data_list = sorted(os.listdir(config["dataset_root"]))
        self.batch_size = config["batch_size"]
        self.patch_size = config["patch_size"]
        self.enhance_hr = config["enhance_hr"]
        self.paired_data = config["paired_data"]
        self.low_resolution = config["low_resolution"]
        self.to_yuv = config["to_yuv"]
        self.sf = config["network"]["scale"]
        # kernel size
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.gaussion_noise_max = 50. / 255.

    def _load_data_list(self, data_list_path):
        with open(data_list_path, "r") as f:
            lines = f.readlines()

        data_list = []
        for line in lines:
            data_item = line.strip("\n ").split(",")
            data_list.append(data_item)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        h_path, l_path = self.data_list[index]

        if "character" in h_path or "pure_color" in h_path:
            self.enhance_hr = False
        else:
            self.enhance_hr = True

        if np.random.uniform() > 0.3 and l_path:
            l_path = l_path
        else:
            l_path = h_path

        img_H = utils_image.img_read(h_path)
        img_L = utils_image.img_read(l_path)

        # resize to low resolution
        if self.low_resolution:
            img_H = utils_image.img_resize(img_H)
            img_L = utils_image.img_resize(img_L)
        
        # augmentation - flip, rotate
        mode = random.randint(0, 7)
        img_H = utils_image.augment_img(img_H, mode=mode)
        img_L = utils_image.augment_img(img_L, mode=mode)
        
        # uint8 to float32
        img_H = utils_image.uint2single(img_H)
        img_L = utils_image.uint2single(img_L)

        # randomly crop a patch
        h, w, _ = img_H.shape
        if self.patch_size % self.sf != 0:
            raise ValueError("patch size is unmatched!")
        l_patch_y = random.randint(0, max(0, h - self.patch_size//self.sf))
        l_patch_x = random.randint(0, max(0, w - self.patch_size//self.sf))
        h_patch_y = l_patch_y * self.sf
        h_patch_x = l_patch_x * self.sf

        patch_H = img_H[h_patch_y:h_patch_y + self.patch_size, h_patch_x:h_patch_x + self.patch_size, :]
        # patch_L = patch_H.copy()
        patch_L = img_L[l_patch_y:l_patch_y + self.patch_size//self.sf, l_patch_x:l_patch_x + self.patch_size//self.sf, :]

        # degradation
        # img_H, img_L = utils_sisr.anime_train_degradation(img_H, self.sf, self.gaussion_noise_max, self.enhance_hr)
        kernel, kernel2, sinc_kernel = utils_sisr.real_degradation(self.opt, self.sf)
        if self.enhance_hr:
        # img_H = utils_image.unsharp_mask(img_H, amount=0.5, threshold=0)
            patch_H = utils_image.usm_sharp(patch_H, weight=0.4, threshold=15)

        # rgb to yuv
        if self.to_yuv:
            patch_H = utils_image.rgb2yuv(patch_H)
            # patch_L = utils_image.rgb2yuv(patch_L)

        # convert to tensor
        patch_H = utils_image.single2tensor(patch_H)
        patch_L = utils_image.single2tensor(patch_L)

        item = {
            "H": patch_H,
            "L": patch_L,
            "kernel": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel
        }

        return item


class TrainDataset(Dataset):

    def __init__(self, config, opt):
        super().__init__()

        self.opt = opt
        self.data_list_path = os.path.join(config["dataset_root"], TRAIN_DATA_LIST_FILENAME)
        self.data_list = self._load_data_list(self.data_list_path)
        # self.data_list = sorted(os.listdir(config["dataset_root"]))
        self.batch_size = config["batch_size"]
        self.patch_size = config["patch_size"]
        self.enhance_hr = config["enhance_hr"]
        self.to_yuv = config["to_yuv"]
        self.sf = config["network"]["scale"]
        # kernel size
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.gaussion_noise_max = 50. / 255.

    def _load_data_list(self, data_list_path):
        with open(data_list_path, "r") as f:
            lines = f.readlines()

        data_list = []
        for line in lines:
            data_item = line.strip("\n ").split(",")
            data_list.append(data_item)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        h_path, l_path = self.data_list[index]
        # h_path = self.data_list[index]
        img_H = utils_image.img_read(h_path)

        # augmentation - flip, rotate
        mode = random.randint(0, 7)
        img_H = utils_image.augment_img(img_H, mode=mode)
        
        # uint8 to float32
        img_H = utils_image.uint2single(img_H)

        # randomly crop a patch
        h, w, _ = img_H.shape
        if self.patch_size % self.sf != 0:
            raise ValueError("patch size is unmatched!")
        l_patch_y = random.randint(0, max(0, h - self.patch_size//self.sf))
        l_patch_x = random.randint(0, max(0, w - self.patch_size//self.sf))
        h_patch_y = l_patch_y * self.sf
        h_patch_x = l_patch_x * self.sf

        patch_H = img_H[h_patch_y:h_patch_y + self.patch_size, h_patch_x:h_patch_x + self.patch_size, :]
        # patch_L = patch_H.copy()
        # patch_L = img_L[l_patch_y:l_patch_y + self.patch_size//self.sf, l_patch_x:l_patch_x + self.patch_size//self.sf, :]

        # degradation
        patch_H, patch_L = utils_sisr.anime_train_degradation(patch_H, self.sf, self.gaussion_noise_max, self.enhance_hr)
        # patch_H, kernel, kernel2, sinc_kernel = utils_sisr.real_degradation(patch_H, self.opt, self.sf, self.enhance_hr)
        

        # rgb to yuv
        if self.to_yuv:
            patch_H = utils_image.rgb2yuv(patch_H)
            # patch_L = utils_image.rgb2yuv(patch_L)

        # convert to tensor
        patch_H = utils_image.single2tensor(patch_H)
        patch_L = utils_image.single2tensor(patch_L)

        item = {
            "H": patch_H,
            "L": patch_L
        }

        return item
