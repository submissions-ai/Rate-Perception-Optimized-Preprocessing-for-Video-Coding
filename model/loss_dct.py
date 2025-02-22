import torch
import torch.nn as nn
import torch.nn.functional
from torch import Tensor
import numpy as np
import math

def blockify(im, size):
    b, c, h, w = im.shape
    
    im = im.reshape(b*c, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    im = im.reshape(b, c, -1, size, size)

    return im

def deblockify(blocks, size):
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks

def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        pass
        # V[:, 0] /= np.sqrt(N) * 2
        # V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    return V

def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        pass
        # X_v[:, 0] *= np.sqrt(N) * 2
        # X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d(X, norm=None):
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def _normalize(N: int) -> Tensor:
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return n @ n.t()


def _harmonics(N: int) -> Tensor:
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(blocks: Tensor) -> Tensor:
    N = blocks.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()

    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff


def block_idct(coeff: Tensor) -> Tensor:
    N = coeff.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


def adaptive_coefficients(coefficients):
    zero = torch.tensor([0], dtype=torch.float32).cuda()
    # print(coefficients.dtype)
    mean_coefficients_value = torch.mean(coefficients.clone(), 3)
    mean_coefficients_value = mean_coefficients_value.reshape(coefficients.shape[0], coefficients.shape[1], coefficients.shape[2], 1)
    # mean_coefficients_value = mean_coefficients_value.long()
    # print(mean_coefficients_value.dtype)
    mean_coefficients = coefficients.clone()
    mean_coefficients[..., :] = mean_coefficients_value
    # print(coefficients.dtype)
    selected_coefficients = torch.where(coefficients < mean_coefficients, zero, coefficients)
    # print(seleceted_coefficients.shape)
    return selected_coefficients

def upper_band_zigzag(coefficients):
    assert len(coefficients.shape) in (4, 5)

    zigzag_indices = torch.tensor([
        #  0,  1,  8, 16,  9,  2,  3, 10,
        # 17, 24, 32, 25, 18, 11,  4,  5,
        # 12, 19, 26, 33, 40, 48, 41, 34,
        # 27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63 
    ]).long() 

    if len(coefficients.shape) == 4:
        c = coefficients.unsqueeze(0)
    else:
        c = coefficients

    # c = blockify(c, 8)

    c = c.view(c.shape[0], c.shape[1], c.shape[2], 64)
    c = c[..., zigzag_indices]
    c = adaptive_coefficients(c)
    # print(c.shape)

    if len(coefficients.shape) == 3:
        c = c.squeeze(0)
    return c

def upper_band_zigzag_4(coefficients):
    assert len(coefficients.shape) in (4, 5)

    zigzag_indices = torch.tensor([
        # 0, 1, 4, 8, 
        # 5, 2, 3, 6, 
        9, 12, 13, 10, 
        7, 11, 14, 15, 
    ]).long() 

    if len(coefficients.shape) == 4:
        c = coefficients.unsqueeze(0)
    else:
        c = coefficients

    # c = blockify(c, 8)

    c = c.view(c.shape[0], c.shape[1], c.shape[2], 16)
    c = c[..., zigzag_indices]
    # c = adaptive_coefficients(c)
    # dynamic_coefficients(c)
    # print(c.shape)

    if len(coefficients.shape) == 3:
        c = c.squeeze(0)
    return c


def upper_band_zigzag_16(coefficients):
    """
    For 16x16 DCT zigzag
    """
    assert len(coefficients.shape) in (4, 5)

    zigzag_indices = torch.tensor([
        # 0, 1, 16, 32, 17, 2, 3, 18, 33, 48, 64, 49, 34, 19, 4, 5, 
        # 20, 35, 50, 65, 80, 96, 81, 66, 51, 36, 21, 6, 7, 22, 37, 52, 
        # 67, 82, 97, 112, 128, 113, 98, 83, 68, 53, 38, 23, 8, 9, 24, 39, 
        # 54, 69, 84, 99, 114, 129, 144, 160, 145, 130, 115, 100, 85, 70, 55, 40, 
        # 25, 10, 11, 26, 41, 56, 71, 86, 101, 116, 131, 146, 161, 176, 192, 177, 
        # 162, 147, 132, 117, 102, 87, 72, 57, 42, 27, 12, 13, 28, 43, 58, 73, 
        # 88, 103, 118, 133, 148, 163, 178, 193, 208, 224, 209, 194, 179, 164, 149, 134, 
        # 119, 104, 89, 74, 59, 44, 29, 14, 15, 30, 45, 60, 75, 90, 105, 120, 
        135, 150, 165, 180, 195, 210, 225, 240, 241, 226, 211, 196, 181, 166, 151, 136, 
        121, 106, 91, 76, 61, 46, 31, 47, 62, 77, 92, 107, 122, 137, 152, 167, 
        182, 197, 212, 227, 242, 243, 228, 213, 198, 183, 168, 153, 138, 123, 108, 93,  
        78, 63, 79, 94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 245, 230, 
        215, 200, 185, 170, 155, 140, 125, 110, 95, 111, 126, 141, 156, 171, 186, 201, 
        216, 231, 246, 247, 232, 217, 202, 187, 172, 157, 142, 127, 143, 158, 173, 188, 
        203, 218, 233, 248, 249, 234, 219, 204, 189, 174, 159, 175, 190, 205, 220, 235,  
        250, 251, 236, 221, 206, 191, 207, 222, 237, 252, 253, 238, 223, 239, 254, 255, 
    ]).long() 

    if len(coefficients.shape) == 4:
        c = coefficients.unsqueeze(0)
    else:
        c = coefficients

    # c = blockify(c, 8)

    c = c.view(c.shape[0], c.shape[1], c.shape[2], 256)
    c = c[..., zigzag_indices]
    # print(c.shape)
    c = adaptive_coefficients(c)

    if len(coefficients.shape) == 3:
        c = c.squeeze(0)
    return c


def lower_band_zigzag(coefficients):
    assert len(coefficients.shape) in (4, 5)

    zigzag_indices = torch.tensor([
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        # 12, 19, 26, 33, 40, 48, 41, 34,
        # 27, 20, 13,  6,  7, 14, 21, 28,
    ]).long() 

    if len(coefficients.shape) == 4:
        c = coefficients.unsqueeze(0)
    else:
        c = coefficients

    c = c.view(c.shape[0], c.shape[1], c.shape[2], 64)
    c = c[..., zigzag_indices]

    if len(coefficients.shape) == 3:
        c = c.squeeze(0)
    return c

class LowerBandEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        blocks = blockify(tensor, 8)
        coffes = block_dct(blocks)
        lower_band = lower_band_zigzag(coffes)
        coffes = torch.abs(lower_band)+1e-8
        probs = torch.nn.functional.normalize(coffes, p=2, dim=3)
        log_prob = torch.log2(probs)
        entropy = torch.mean(-torch.sum(probs*log_prob, dim=3))
 
        return entropy

class UpperBandEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        blocks = blockify(tensor, 8)
        coffes = block_dct(blocks)
        upper_band = upper_band_zigzag(coffes)
        coffes = torch.abs(upper_band)+1e-12
        probs = torch.nn.functional.normalize(coffes, p=2, dim=3)
        log_prob = torch.log2(probs)
        entropy = torch.mean(-torch.sum(probs*log_prob, dim=3))
 
        return entropy

class UpperBandEntropyLoss16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        blocks = blockify(tensor, 16)
        coffes = block_dct(blocks)
        upper_band = upper_band_zigzag_16(coffes)
        coffes = torch.abs(upper_band)+1e-12
        probs = torch.nn.functional.normalize(coffes, p=2, dim=3)
        log_prob = torch.log2(probs)
        entropy = torch.mean(-torch.sum(probs*log_prob, dim=3))
 
        return entropy

class LowRankLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        blocks = blockify(tensor, 8)
        coffes = block_dct(blocks)
        upper_band = upper_band_zigzag(coffes)
        l2 = torch.mean(torch.sum(torch.pow(upper_band, 2), dim=3))

        return l2

class LowRankLoss16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        blocks = blockify(tensor, 16)
        coffes = block_dct(blocks)
        upper_band = upper_band_zigzag_16(coffes)
        l2 = torch.mean(torch.sum(torch.pow(upper_band, 2), dim=3))

        return l2

class LowRankLoss4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        blocks = blockify(tensor, 4)
        coffes = block_dct(blocks)
        upper_band = upper_band_zigzag_4(coffes)
        l2 = torch.mean(torch.sum(torch.pow(upper_band, 2), dim=3))

        return l2
        
if __name__ == "__main__":
    mat = np.array(list(range(0, 32*32))).reshape([1, 1, 32, 32]).astype(np.float32)
    tensor = torch.from_numpy(mat)
    blocks = blockify(tensor, 8)
    coffes = block_dct(blocks)

    blocks = block_idct(coffes)
    result = deblockify(blocks, [32, 32])
    print(result.numpy().flatten().tolist()) 