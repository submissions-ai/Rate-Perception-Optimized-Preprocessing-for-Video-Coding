import torch as t
import torch.nn as nn
import torch.nn.functional as nf
import torchvision.transforms.functional as tf
import torchvision.models

class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16.eval()
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_norm = tf.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        y_norm = tf.normalize(y, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        x_feature = self.vgg16.features[:16](x_norm) # 只用到block_conv3_3
        y_feature = self.vgg16.features[:16](y_norm)

        feature_mse = nf.mse_loss(x_feature, y_feature)
        return feature_mse

class SobelLoss(nn.Module):

    def __init__(self):
        super().__init__()

        # sobel filters
        self.sobel_kernel_y = nn.Parameter(t.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=t.float32).expand(1, 3, 3, 3), False)
        self.sobel_kernel_x = nn.Parameter(t.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=t.float32).expand(1, 3, 3, 3), False)

    def _sobel(self, x):
        assert x.size()[1] == 3 # channel == 3
        y_grad = nf.conv2d(x, self.sobel_kernel_y)
        x_grad = nf.conv2d(x, self.sobel_kernel_x)
        sobel = t.cat([y_grad, x_grad], dim=1)
        return sobel

    def forward(self, x, y):
        x_sobel = self._sobel(x)
        y_sobel = self._sobel(y)
        sobel_loss = t.mean(t.abs(x_sobel - y_sobel))
        return sobel_loss

class RFDNLoss(t.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.l1_loss = t.nn.SmoothL1Loss()
        if config["loss"] is not None and "perceptual" in config["loss"] :
            self.perceptual_loss = PerceptualLoss()
            self.use_perceptual = True
        else:
            self.perceptual_loss = 0
            self.use_perceptual = False
        if config["loss"] is not None and "sobel" in config["loss"] :
            self.sobel_loss = SobelLoss()
            self.use_sobel = True
        else:
            self.sobel_loss = 0
            self.use_sobel = False
        
        self.l1_weight = 1.0
        self.perceptual_weight = 0.1
        self.sobel_weight = 1.0

    def forward(self, x, y):
        l1 = self.l1_loss(x, y)
        p = self.perceptual_loss(x, y) if self.use_perceptual else 0
        s = self.sobel_loss(x, y) if self.use_sobel else 0

        l = l1 * self.l1_weight + p * self.perceptual_weight + s * self.sobel_weight
        return l
