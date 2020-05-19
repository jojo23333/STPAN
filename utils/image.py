import os, cv2
import numpy as np
import torch

def sRGBforward(x):
    b = .0031308
    gamma = 1./2.2
    # a = .055
    # k0 = 12.92
    a = 1./(1./(b**gamma*(1.-gamma))-1.)
    k0 = (1+a)*gamma*b**(gamma-1.)
    gammafn = lambda x : (1+a)*tf.pow(tf.maximum(x,b),gamma)-a
    # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
    srgb = tf.where(x < b, k0*x, gammafn(x))
    k1 = (1+a)*gamma
    srgb = tf.where(x > 1, k1*x-k1+1, srgb)
    return srgb


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


def uint2tensor4(img):
    assert img.ndim == 4, "uint2tensor4 input must be batch of images, get {}".format(img.shape)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).float().div(255.)


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    elif img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def get_split_indexes(H, W, p_size):
    indexs = []
    h1 = 0
    while h1 < H:
        h = H - p_size if h1 + p_size > H else h1
        w1 = 0
        while w1 < W:
            w = W - p_size if w1 + p_size > W else w1
            indexs.append((h, w))
            w1 = w1 + p_size - 5
        h1 = h1 + p_size -5
    return indexs

def split_test(img, test_patch_size=512):
    B, T, C, H, W = img.shape
    P_SIZE = test_patch_size
    assert B == 1
    batch = []
    indexes = get_split_indexes(H, W, P_SIZE)
    
    for h, w in indexes:
        batch.append(img[0:1, :, :, h:h+P_SIZE, w:w+P_SIZE])
    return batch

def merge_test(imgs, gt, test_patch_size=512):
    _, C, H, W = gt.shape
    print("{} {}".format(H, W))
    P_SIZE = test_patch_size
    indexes = get_split_indexes(H, W, P_SIZE)
    mask = torch.zeros((1, 1, H, W))
    merged_img = torch.zeros((1, C, H, W))
    padding = 2
    for i, (h, w) in enumerate(indexes):
        h1_padding = 0 if h == 0 else padding
        h2_padding = 0 if h + P_SIZE == H else padding
        w1_padding = 0 if w == 0 else padding
        w2_padding = 0 if w + P_SIZE == W else padding
        h1 = h + h1_padding
        h2 = h + P_SIZE - h2_padding
        w1 = w + w1_padding
        w2 = w + P_SIZE - w2_padding

        merged_img[0, :, h1:h2, w1:w2] += imgs[i][0, :, h1_padding:P_SIZE-h2_padding, w1_padding:P_SIZE-w2_padding]
        mask[0, :, h1:h2, w1:w2] += 1.0
    merged_img = merged_img / mask
    return merged_img


# def imread_tensor3(path, n_channels=3):
#     return uint2tensor3(imread_uint(path, n_channels))
