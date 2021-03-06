#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.videoDenoiseDataset import videoDenoiseDataset
from datasets.vimeo import get_vimeo20k_instances

# Supported datasets.
_DATASET_CATALOG = { 
    "vimeo20k_video" : videoDenoiseDataset,
}

_DATASET_COLLECTOR_CATALOG = {
    "vimeo20k_video" : get_vimeo18k_instances
}


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = True
        drop_last = True
        num_workers = cfg.DATA_LOADER.NUM_WORKERS
    elif split in ["val"]:
        dataset_name = cfg.VAL.DATASET
        batch_size = 1
        shuffle = False
        drop_last = False
        num_workers = 0
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = False
        drop_last = False
        num_workers = cfg.DATA_LOADER.NUM_WORKERS
    assert (
        dataset_name in _DATASET_CATALOG.keys()
    ), "Dataset '{}' is not supported".format(dataset_name)

    get_instances_list =  lambda :_DATASET_COLLECTOR_CATALOG[dataset_name](cfg, split)
    # Construct the dataset
    dataset = _DATASET_CATALOG[dataset_name](cfg, split, get_instances_list)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
