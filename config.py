import os
import yaml
### Lowrank 8 weight 8 & Lowrank 16 weight 8 & usm_sharp(img_H, weight=0.4, threshold=15)
### Model RFLite_v2
DEFAULT_CONFIG_PATH = "./configs/default.yaml"
DEFAULT_CONFIG = {
    "degradation_type": "paired", # real or paired or norm
    "prefix": "preprocess_v2",
    "manual_seed": None,
    # "dataset_list_path": "/data/docker/machengqian/dataset/flicker_div_ffhq_data_patch_list.txt",
    "dataset_list_path": "/data/docker/machengqian/dataset/flicker_div_ffhq_anime_pubg_genshin_danbooru_character_pdf_dense_dark_data_patch_v13_1000k.txt",
    "val_dataset_path": "/data/docker/machengqian/dataset/Flicker_Div_combined_dataset",
    "batch_size": 32,
    "num_workers": 32,
    "patch_size": 64,
    "paired_data": False,
    "enhance_hr": True,
    "low_resolution": False,
    "to_yuv": False,
    "network": {
        "scale": 1,
        "pretrained": "/data/docker/machengqian/preprocess/preprocess-rembrandt/checkpoints/step_2580000.pth",
    },
    "loss": "",
    "gpus": [0, 1],
    "learning_rate": 0.0001,
    "scheduler_milestones": [200, 600, 1000],
    "scheduler_gamma": 0.5,
    "steps_val": 20000,
    "output_dir": "output/20220707_v62_4",
    "dist": True
}
DEGRADATION_CONFIG = {
    "resize_prob": [0.3, 0.5, 0.2],  # up, down, keep
    "resize_range": [0.5, 1.5],
    "gaussian_noise_prob": 0.5,
    "noise_range": [1, 2],
    "poisson_scale_range": [0.05, 0.5],
    "gray_noise_prob": 0.3,
    "jpeg_range": [30, 95],
    "skip_degradation_prob": 0.2, 
    
# the second degradation process
    "second_blur_prob": 0,
    "resize_prob2": [0.3, 0.4, 0.3],  # up, down, keep
    "resize_range2": [0.3, 1.2],
    "gaussian_noise_prob2": 0,
    "noise_range2": [1, 5],
    "poisson_scale_range2": [0.05, 0.5],
    "gray_noise_prob2": 0,
    "jpeg_range2": [80, 95],

    "do_first_blur": False,
    "blur_kernel_size": 21,
    "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob": 0.2,
    "blur_sigma": [0.2, 0.5],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],

    "blur_kernel_size2": 21,
    "kernel_list2": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob2": 0,
    "blur_sigma2": [0.2, 0.5],
    "betag_range2": [0.5, 4],
    "betap_range2": [1, 2],
    "scale": 1,
    "final_sinc_prob": 0
}

def load_config(path=DEFAULT_CONFIG_PATH):
    try:
        with open(path, "r") as f:
            config_data = yaml.load(f, Loader=yaml.SafeLoader)
            config_data["learning_rate"] = float(config_data["learning_rate"])
        return config_data
    except:
        return DEFAULT_CONFIG
