import sys
sys.path.append("/data/docker/machengqian/preprocess/preprocess-rembrandt")
import copy
import numpy as np
import torch
import cv2
from torch import nn
import torchvision.transforms.functional as tvf
from model.preprocess_rfdn import RFLite_v2, RFLite_v3, RFLite_v5
from utils.misc import *


class RGB2YUV(nn.Module):
    def __init__(self):
        super(RGB2YUV, self).__init__()
    
    def forward(self, x):
        '''
        Input:
            torch.float32, [0, 1]
            shape [c, h, w], rgb
        '''
        assert x.dtype == torch.float32
        m = torch.tensor([[0.2989, -0.1688, 0.5000],
                    [0.5866, -0.3312, -0.4184],
                    [0.1145, 0.5000, -0.0816]], dtype=torch.float32)
        rgb = x.permute(0, 2, 3, 1)
        yuv = torch.matmul(rgb, m)
        yuv[:, :, :, 0] = (yuv[:, :, :, 0] * 219. + 16.) / 255.
        yuv[:, :, :, 1] = (yuv[:, :, :, 1] * 224. + 128.) / 255.
        yuv[:, :, :, 2] = (yuv[:, :, :, 2] * 224. + 128.) / 255.
        x = yuv.permute(0, 3, 1, 2)

        return x

class YUV2RGB(nn.Module):
    def __init__(self):
        super(YUV2RGB, self).__init__()
    
    def forward(self, x):
        '''
        Input:
            torch.float32, [0, 1]
            shape [c, h, w], yuv
        '''
        assert x.dtype == torch.float32
        yuv = x.permute(0, 2, 3, 1)
        yuv[:, :, :, 0] = (yuv[:, :, :, 0] * 255. - 16.) / 219.
        yuv[:, :, :, 1] = (yuv[:, :, :, 1] * 255. - 128.) / 224.
        yuv[:, :, :, 2] = (yuv[:, :, :, 2] * 255. - 128.) / 224.
        m = torch.tensor([[1.0, 1.0, 1.0],
                    [0, -0.3456, 1.7710],
                    [1.4022, -0.7145, 0]], dtype=torch.float32)
        rgb = torch.matmul(yuv, m)    
        x = rgb.permute(0, 3, 1, 2)

        return x

class WOP(nn.Module):
    def __init__(self):
        super(WOP, self).__init__()
        self.yuv_to_rgb = YUV2RGB()
        self.rgb_to_yuv = RGB2YUV()

    def forward(self, x):
        rgb = self.yuv_to_rgb(x)
        yuv = self.rgb_to_yuv(rgb)
        return yuv

class WOP_RYR(nn.Module):
    def __init__(self):
        super(WOP_RYR, self).__init__()
        self.yuv_to_rgb = YUV2RGB()
        self.rgb_to_yuv = RGB2YUV()

    def forward(self, x):
        yuv = self.rgb_to_yuv(x)
        rgb = self.yuv_to_rgb(yuv)
        return rgb

if __name__ == "__main__":

    checkpoint_path = "/data/docker/machengqian/preprocess/preprocess-rembrandt/checkpoints/step_3600000.pth"
    # checkpoint_path = "/data/docker/machengqian/preprocess-mmai/scripts/rflitev3yuv_1015400_0.01488660.pth"
    device = "cpu"
    # precision = torch.float32
    precision = torch.uint8
    # save_path = "./rflitev3_20220311_step_1015400_yuv.onnx"
    # save_path = "./rflitv2_20220408_v39_3_hwc_255_uint8.onnx"
    save_path = "./rflitv2_20220711_v68_3_hwc_255_uint8_dynamic.onnx"
    model = RFLite_v5()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"], strict=False)
    # img = cv2.imread("/data/docker/machengqian/dataset/Flicker_Div_combined_dataset/0001.png")
    # ori = copy.deepcopy(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.float32(img/255.)
    # tensor = tvf.to_tensor(img)
    # tensor = torch.from_numpy(img)
    src = torch.randn(1, 1080, 1920, 3).to(device, precision)
    # src = torch.randn(1, 3, 1080, 1920).to(device, precision)
    
    # model = WOP_RYR()

    # out = model(tensor.unsqueeze(0))
    # out = out.data.squeeze().float().cpu().numpy()
    # out = np.transpose(out, (1, 2, 0))
    # out = np.uint8((out.clip(0, 1)*255.).round())
    # img = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # res = ori - img
    # print(np.mean(res))

    # cv2.imwrite("./test.png", img)

    # src = torch.randn(1, 3, 360, 640).to(device, precision)

    # torch.onnx.export(
    #         model,
    #         src,
    #         save_path,
    #         export_params=True,
    #         opset_version=12,
    #         do_constant_folding=True,
    #         input_names=['input'],
    #         output_names=['output']
    #         )
    
    torch.onnx.export(
            model,
            src,
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes={'input':{1:'height', 2:'width'},'output':{1:'height', 2:'width'}},
            input_names=['input'],
            output_names=['output']
            )


