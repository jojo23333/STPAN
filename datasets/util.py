import os, cv2
import numpy as np
import torch

import logging

class imageCorupptedError(Exception):
    def __init__(self):
        super(imageCorupptedError, self).__init__()

def get_image_subpaths(source_root, t_sample_size=5, mode='train'):
    frame_patches = []
    all_vids = os.listdir(source_root)
    val_vids = ['008', '029', '034', '035']
    train_vids = list(set(all_vids) - set(val_vids))
    if mode == 'train':
        vids = train_vids
    elif mode == 'val':
        vids = val_vids
    elif mode == 'test':
        vids = all_vids

    for vid in vids:
        frames = os.listdir(os.path.join(source_root, vid))
        frames.sort()
        padding = t_sample_size//2 
        for i in range(padding, len(frames) - padding):
            if mode == 'val' and i % 20 != 9:
                continue
            if mode == 'test' and i % 20 != 19:
                continue
            patch = [os.path.join(vid, frames[j]) for j in range(i-padding, i+padding+1)]
            frame_patches.append(patch)
    
    return frame_patches

def get_test_image_subpaths(source_root, t_sample_size=5, mode='part'):
    frame_patches = []
    all_vids = os.listdir(source_root)
    all_vids.sort()

    for vid in all_vids:
        frames = os.listdir(os.path.join(source_root, vid))
        frames.sort()
        padding = t_sample_size//2 
        for i in range(0, len(frames)):
            if mode == 'part' and i % 20 != 19:
                continue
            patch = []
            if padding - i > 0:
                patch = patch + [os.path.join(vid, frames[i]) for x in range(0, padding - i)]
            begin = max(0, i-padding)
            end = min(len(frames), i+padding+1)
            patch = patch + [os.path.join(vid, frames[j]) for j in range(begin, end)]
            if padding + i + 1 - len(frames)> 0:
                patch = patch + [os.path.join(vid, frames[i]) for x in range(0, padding + i + 1 - len(frames))]
            frame_patches.append(patch)
    
    return frame_patches

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def augment_frames(frames, mode=0):    
    aug_frames = []
    b, h, w, c = frames.shape
    for i in range(b):
        aug_frames.append(augment_img(frames[i,...], mode))
    return np.stack(aug_frames, axis=0)

# get uint8 image of size HxWxn_channles (RGB)
def read_img(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img is None:
            logging.warning("{} do not exsist!".format(path))
            raise imageCorupptedError
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

###################### read images ######################
def read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


# def read_img(env, path, size=None):
#     """read image by cv2 or from lmdb
#     return: Numpy float32, HWC, BGR, [0,1]"""
#     if env is None:  # img
#         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     else:
#         img = _read_img_lmdb(env, path, size)
#     img = img.astype(np.float32) / 255.
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=2)
#     # some images have 4 channels
#     if img.shape[2] > 3:
#         img = img[:, :, :3]
#     return img


def read_img_seq(path):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
    img_l = [read_img(None, v) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

if __name__ == "__main__":
    a = get_image_subpaths("/Disk2/limuchen/NITRE2020/train/source")    
    print(len(a), a[0])