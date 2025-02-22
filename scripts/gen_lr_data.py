import sys, os, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import cv2
from utils import utils_image, utils_sisr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, default="./hr_dir", help="the folder of the input HR images")
    parser.add_argument("--lr_dir", type=str, default="./lr_dir", help="the folder of the output LR images")
    parser.add_argument("--is_val", action="store_true", help="apply val data degradation model")
    opt = parser.parse_args()

    hr_dir = opt.hr_dir
    lr_dir = opt.lr_dir
    is_val = opt.is_val

    sf = 2
    gaussion_noise_max = 20. / 255.
    enhance_hr = False

    hr_imgs = utils_image.get_image_paths(hr_dir)
    for hr_path in hr_imgs:
        img_basename = os.path.basename(hr_path)

        img_H = utils_image.img_read(hr_path)
        img_H = utils_image.uint2single(img_H)

        if is_val:
            _, img_L = utils_sisr.anime_val_degradation(img_H, sf, gaussion_noise_max, enhance_hr)
        else:
            _, img_L = utils_sisr.anime_train_degradation(img_H, sf, gaussion_noise_max, enhance_hr)

        img_L = utils_image.single2uint(img_L)
        lr_path = os.path.join(lr_dir, img_basename)
        utils_image.img_save(img_L, lr_path)

        print(f"{img_basename}")
        
if __name__ == "__main__":
    main()