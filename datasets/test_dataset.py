import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from utils import utils_image, utils_sisr

class TestDataset(Dataset):

    def __init__(self, data_dir, to_yuv=True):
        super().__init__()

        self.data_dir = data_dir
        self.data_list = self._load_data_list(self.data_dir)
        self.to_yuv = to_yuv
        self.gaussion_noise_max = 20. / 255.
        self.enhance_hr = False

    def _load_data_list(self, data_dir):
        img_paths = utils_image.get_image_paths(data_dir)
        
        data_list = img_paths
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        l_path = self.data_list[index]
        img_O = utils_image.img_read(l_path)
        # padding to 8 scale
        img_L = utils_image.img_padding(img_O.copy())
        img_L = utils_image.uint2single(img_L)
        img_O = utils_image.uint2single(img_O)

        # img_H, img_L = utils_sisr.anime_val_degradation(img_L, 1, self.gaussion_noise_max, self.enhance_hr)
        img_H, img_L = utils_sisr.anime_train_degradation(img_L, 1, self.gaussion_noise_max, self.enhance_hr)
        # img_H = img_O
        # if self.to_yuv:
        #     img_L = utils_image.rgb2yuv(img_L)
        #     img_H = utils_image.rgb2yuv(img_H)

        img_L = tf.to_tensor(img_L)
        img_H = tf.to_tensor(img_H)

        item = img_L, img_H, l_path
        return item
