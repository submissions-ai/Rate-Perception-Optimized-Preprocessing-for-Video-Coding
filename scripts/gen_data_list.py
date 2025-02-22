import sys, os, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils import data
import argparse
from utils import utils_image

out_filename = "character_pdf_dense_patch.txt" 

def write_data_list(data_list, file_path):
    all_content = ""
    for data_item in sorted(data_list):
        if len(data_item) == 2:
            s = f"{data_item[0]},{data_item[1]}\n"
        else:
            s = f"{data_item[0]}, \n"
        all_content += s
    with open(file_path, "w") as f:
        f.write(all_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, default="./hr_dir", help="the folder of the HR images")
    parser.add_argument("--lr_dirs", nargs="*", default=[""], help="the list of the LR image folders")
    parser.add_argument("--paired",  action="store_true", help="only hr-lr paired data")
    opt = parser.parse_args()

    hr_dir = opt.hr_dir
    lr_dirs = opt.lr_dirs
    paired = opt.paired

    data_list_with_lr = []
    # for lr_dir in lr_dirs:
    #     lr_imgs = utils_image.get_image_paths(lr_dir)
    #     for lr_img_path in lr_imgs:
    #         img_filename = os.path.basename(lr_img_path)
    #         hr_img_filename = img_filename.replace("jpg", "png")
    #         hr_img_path = os.path.join(hr_dir, hr_img_filename)
    #         # print(hr_img_path)
    #         if os.path.exists(hr_img_path):
    #             data_item = hr_img_path, lr_img_path
    #             data_list_with_lr.append(data_item)

    data_list_only_hr = []
    hr_imgs = utils_image.get_image_paths(hr_dir)
    for hr_img_path in hr_imgs:
        data_item = hr_img_path, 
        data_list_only_hr.append(data_item)

    if paired:
        all_data_list = data_list_with_lr
    else:
        all_data_list = data_list_with_lr + data_list_only_hr

    write_data_list(all_data_list, out_filename)

if __name__ == "__main__":
    main()
