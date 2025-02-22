import sys, os, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
from utils import utils_image

PATCH_SIZE = 512

def generate_patches(img, norm=True):
    h, w, c = img.shape
    if h < PATCH_SIZE or w < PATCH_SIZE:
        return []

    patch_num = (h // PATCH_SIZE + 1) * (w // PATCH_SIZE + 1)

    patches = []
    for i in range(patch_num):
        if norm:
            y = int(max(0, min(((np.random.randn() + 3.0) / 6.0) * h, h - PATCH_SIZE)))
            x = int(max(0, min(((np.random.randn() + 3.0) / 6.0) * w, w - PATCH_SIZE)))
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]
        else:
            y = random.randint(0, max(0, h - PATCH_SIZE))
            x = random.randint(0, max(0, w - PATCH_SIZE))
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]
        patches.append(patch)
    return patches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="./hr_images", help="the folder of the HR images")
    parser.add_argument("--out_dir", type=str, default="./hr_patches", help="the folder of the HR patches")
    parser.add_argument("--norm", action="store_true", help="normal distribution sample")
    opt = parser.parse_args()

    in_dir = opt.in_dir
    out_dir = opt.out_dir
    norm = opt.norm

    imgs = utils_image.get_image_paths(in_dir)
    for i, img_path in enumerate(imgs):
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        category = os.path.basename(os.path.dirname(img_path))
        img = utils_image.img_read(img_path)
        try:
            patches = generate_patches(img, norm)
            for j, patch in enumerate(patches):
                patch_path = os.path.join(out_dir, f"{category}_{img_basename}_p{j}.png")
                utils_image.img_save(patch, patch_path)
        except:
            print(f"error path: {img_path}")
        sys.stdout.write(f"\r[{i+1}/{len(imgs)}]")

    print("Done.")

if __name__ == "__main__":
    main()
