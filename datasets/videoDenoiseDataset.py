import torch
import random
import lmdb
import os
import numpy as np
import utils.image as im
import utils.logging as logging

from os.path import join as Join
from fvcore.common.timer import Timer
from datasets.util import *

def VideoNoisemapper(cfg, source_frames):
    SP_PATCH_SIZE = cfg.DATA.SP_PATCH_SIZE
    NOISE_TYPE = cfg.DENOISE.NOISE_TYPE
    SIGMA = cfg.DENOISE.GAUSSIAN_SIGMA
    MID_FRAME = cfg.DENOISE.NUM_FRAMES // 2

    _, H, W, C = source_frames.shape
    rand_h = random.randint(0, max(0, H - SP_PATCH_SIZE))
    rand_w = random.randint(0, max(0, W - SP_PATCH_SIZE))
    source_frames = source_frames[:, rand_h:rand_h+SP_PATCH_SIZE, rand_w:rand_w+SP_PATCH_SIZE, :]
    mode = np.random.randint(0, 8)
    source_frames = augment_frames(source_frames, mode)
    source_frames = im.uint2tensor4(source_frames)

    target = source_frames[MID_FRAME, ...]
    if NOISE_TYPE == "gaussian":
        noise = torch.randn_like(source_frames, dtype=torch.float32) * SIGMA / 255.
        noisy_frames = noise + source_frames
        noisy_frames.clamp(0, 1)
        noise_level = SIGMA
    else:
        source_frames = source_frames ** 2.2
        sig_read = torch.pow(10, torch.empty([1, 1, 1, 1]).random_(-3., -1.5))
        sig_shot = torch.pow(10, torch.empty([1, 1, 1, 1]).random_(-2., -1))
        read_noise = sig_read * torch.randn_like(source_frames)
        shot_noise = sig_shot * torch.sqrt(source_frames) * torch.randn_like(source_frames)
        noisy_frames = read_noise + shot_noise + source_frames 
        noisy_frames.clamp(0, 1)
        noise_level = (sig_read.item(), sig_shot.item())#torch.sqrt(sig_read ** 2 + sig_shot ** 2 * noisy_frames[MID_FRAME, ...])
    return {"input": noisy_frames, "gt": target, "noise_level": noise_level}
        

class videoDenoiseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, get_instances_list):
        super(NtireQualityMapping, self).__init__()
        self.mode = mode
        self.instances = get_instances_list()
        self.mapper = lambda x: VideoNoisemapper(cfg, x)
        self.SP_PATCH_SIZE = cfg.DENOISE.SP_PATCH_SIZE
        self.MID_FRAME = cfg.DENOISE.NUM_FRAMES // 2
        self.NOISE_TYPE = cfg.DENOISE.NOISE_TYPE
        self.TEST_SIGMA = cfg.DENOISE.TEST_SIGMA if self.NOISE_TYPE == "gaussian" else (cfg.DENOISE.TEST_SIG_READ, cfg.DENOISE.TEST_SIG_SHOT)

    def __getitem__(self, index):
        patch = self.instances[index]
        if self.mode == 'train':
            # if target_img do not exists random_select one
            while True:
                try:
                    source_frames = []
                    for img_path in patch:
                        source_frames.append(read_img(img_path))
                except imageCorupptedError:
                    index = random.randint(0, len(self.frame_patches) - 1)
                    patch = self.frame_patches[index]
                else:
                    break
            source_frames = np.stack(source_frames, axis=0)
            datadict = self.mapper(source_frames)
            return datadict
        else:
            noisy_patch, gt_patch = patch
            noisy_frames = []
            for img_path in noisy_patch:
                noisy_frames.append(read_img(img_path))
            gt_frame = read_img(gt_patch[self.MID_FRAME])
            gt_img_path = gt_patch[self.MID_FRAME]
            return {"input": source_frames, "gt": patch, "noise_level": self.TEST_SIGMA}
        
    
    def __len__(self):
        return len(self.instances)

