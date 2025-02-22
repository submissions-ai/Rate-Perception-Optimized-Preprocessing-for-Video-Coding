import os, sys, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
import torch as t
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from utils import utils_image, utils_sisr

VAL_DATA_LIST_FILENAME = "flicker_div_data_list.txt"

class ValDataset(Dataset):

    def __init__(self, config):
        super().__init__()

        self.data_list_path = os.path.join(config["val_dataset_path"], VAL_DATA_LIST_FILENAME)
        self.data_list = self._load_data_list(self.data_list_path)
        # self.data_list = sorted(os.listdir(config["dataset_root"]))
        self.to_yuv = config["to_yuv"]
        self.enhance_hr = config["enhance_hr"]
        self.sf = config["network"]["scale"]
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
        # padding image to 8 scale
        img_H = utils_image.img_padding(img_H)
        # uint8 to float32
        img_H = utils_image.uint2single(img_H)

        # degradation
        # img_H, img_L = utils_sisr.anime_val_degradation(img_H, self.sf, self.gaussion_noise_max, self.enhance_hr)
        img_H, img_L = utils_sisr.anime_train_degradation(img_H, self.sf, self.gaussion_noise_max, self.enhance_hr)
        assert img_H.shape == img_L.shape
        # rgb to yuv
        if self.to_yuv:
            img_H = utils_image.rgb2yuv(img_H)
            img_L = utils_image.rgb2yuv(img_L)

        # convert to tensor
        img_H = tf.to_tensor(img_H)
        img_L = tf.to_tensor(img_L)

        item = {
            "H": img_H,
            "L": img_L
        }

        return item

class ValDataset_real(Dataset):

    def __init__(self, config, opt):
        super().__init__()

        self.opt = opt
        self.data_list_path = os.path.join(config["val_dataset_path"], VAL_DATA_LIST_FILENAME)
        self.data_list = self._load_data_list(self.data_list_path)
        # self.data_list = sorted(os.listdir(config["dataset_root"]))
        self.to_yuv = config["to_yuv"]
        self.enhance_hr = config["enhance_hr"]
        self.sf = config["network"]["scale"]
        self.gaussion_noise_max = 20. / 255.

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
        # padding image to 8 scale
        img_H = utils_image.img_padding(img_H)
        # uint8 to float32
        img_H = utils_image.uint2single(img_H)
        img_L = img_H.copy()
        # degradation
        # img_H, img_L = utils_sisr.anime_val_degradation(img_H, self.sf, self.gaussion_noise_max, self.enhance_hr)
        # img_H, img_L = utils_sisr.anime_train_degradation(img_H, self.sf, self.gaussion_noise_max, self.enhance_hr)
        kernel, kernel2, sinc_kernel = utils_sisr.real_degradation(self.opt, self.sf)


        assert img_H.shape == img_L.shape
        # rgb to yuv
        if self.to_yuv:
            img_H = utils_image.rgb2yuv(img_H)
            img_L = utils_image.rgb2yuv(img_L)

        # convert to tensor
        img_H = tf.to_tensor(img_H)
        img_L = tf.to_tensor(img_L)

        item = {
            "H": img_H,
            "L": img_L,
            "kernel": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel
        }

        return item