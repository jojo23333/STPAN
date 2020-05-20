#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from utils.image import split_test, merge_test, tensor2uint

import models.optimizer as optim
import models.losses as losses
import models.model_builder as model_builder

import utils.metrics as metrics
import utils.misc as misc
import utils.distributed as du
import utils.checkpoint as cu
import utils.logging as logging

import datasets.loader as loader

from utils.dct import get_dct_mask_at_epoch, get_dct_mask_at_validation_epoch
from utils.meters import TrainMeter, ValMeter


logger = logging.get_logger('default')


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cur_gs, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    gs = cur_gs
 
    # Update the learning rate.
    lr = optim.get_epoch_lr(cur_epoch, cfg)
    optim.set_lr(optimizer, lr)

    for cur_iter, (inputs, gt) in enumerate(train_loader):
        # logger.info('toc')
        gs = gs + 1
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        gt = gt.cuda()
        losses, training_infos = model(inputs, gt, gs)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        train_meter.iter_toc()
        # Update and log stats.
        training_infos.update(losses)
        training_infos.update({"lr": lr})
        # log info into meters
        train_meter.update_stats(training_infos)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

        # Update global step, lr, and multigrid stages
        gs = gs + 1
        # if cur_iter % 100 == 99:
        #     cu.save_checkpoint_iter(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, gs, cur_iter, cfg)

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return gs


@torch.no_grad()
def validation_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
        Evaluate model
    """
    model.eval()
    logger.info("Start validation")

    if 'dct' in cfg.MODEL.ARCH and cfg.SOLVER.USE_DCT_MASK:
        freq_mask = get_dct_mask_at_validation_epoch(cfg, cur_epoch).cuda()
    else:
        freq_mask = None

    # batch_size is fixed to 1 on each GPU
    for inputs, gt, frame_ids in val_loader:
        batch = split_test(inputs, cfg.VAL.VAL_PATCH_SIZE)
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        print(inputs.shape, flush=True)
        preds = []
        for inputs in batch:
            if freq_mask is not None:
                output = model(inputs, freq_mask)
            else:
                output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            preds.append(output.cpu())
        pred = merge_test(preds, gt, cfg.VAL.VAL_PATCH_SIZE)

        for i, fid in enumerate(frame_ids):
            vid, img_id = fid.split('/')
            pred = tensor2uint(pred[i, ...])
            psnr, ssim = -1, -1
            if gt is not None:
                gt = tensor2uint(gt[i, ...])
                ssim = metrics.calculate_ssim(pred, gt)
                psnr = metrics.calculate_psnr(pred, gt)

            # save img
            # save statistic of each img
            # if cfg.NUM_GPUS > 1:
            #     ssim = du.all_gather(ssim)
            #     psnr = du.all_gather(psnr)
            #     vid = du.all_gather(vid)
            #     img_id = du.all_gather(img_id)
            #     for i in range(cfg.NUM_GPUS):
            #         val_meter.log_img_result(vid[i], img_id[i], psnr[i], ssim[i])
            # else:
            val_meter.log_img_result(vid, img_id, psnr ,ssim)

    val_meter.log_average_score(cur_epoch)
    val_meter.reset()

    return


@torch.no_grad()
def validation_epoch_center(val_loader, model, val_meter, cur_epoch, cfg):
    """
        Evaluate model
    """
    model.eval()
    logger.info("Start validation")

    if 'dct' in cfg.MODEL.ARCH and cfg.SOLVER.USE_DCT_MASK:
        freq_mask = get_dct_mask_at_validation_epoch(cfg, cur_epoch).cuda()
    else:
        freq_mask = None

    # batch_size is fixed to 1 on each GPU
    for inputs, gt, frame_ids in val_loader:
        _, _, C, H, W = inputs.size()
        inputs = inputs[:, :, :, H//2-320:H//2+320, W//2-320:W//2+320]
        gt = gt[:, :, H//2-320:H//2+320, W//2-320:W//2+320]
        # Transfer the data to the current GPU device.
        inputs = inputs.contiguous()
        gt = gt.contiguous()
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        print(inputs.shape, flush=True)
        
        if freq_mask is not None:
            pred = model(inputs, freq_mask)
        else:
            pred = model(inputs)
        # if isinstance(output, tuple):
        #     output = output[0]

        for i, fid in enumerate(frame_ids):
            vid, img_id = fid.split('/')
            pred = tensor2uint(pred[i, ...])
            psnr, ssim = -1, -1
            if gt is not None:
                gt = tensor2uint(gt[i, ...])
                ssim = metrics.calculate_ssim(pred, gt)
                psnr = metrics.calculate_psnr(pred, gt)

            val_meter.log_img_result(vid, img_id, psnr ,ssim)

    val_meter.log_average_score(cur_epoch)
    val_meter.reset()
    return



def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Setup logging format.
    logging.setup_logging(logger, cfg)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Record global step
    gs = 0

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        gs, checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        if cfg.TRAIN.LOAD_PART_OF_CHECKPOINT:
            gs, checkpoint_epoch = cu.load_part_of_checkpoint(
                cfg.TRAIN.CHECKPOINT_FILE_PATH,
                model,
                cfg.NUM_GPUS > 1,
                optimizer=None
            )
        else:
            gs, checkpoint_epoch = cu.load_checkpoint(
                cfg.TRAIN.CHECKPOINT_FILE_PATH,
                model,
                cfg.NUM_GPUS > 1,
                optimizer=None,
                inflation=False,
                convert_from_caffe2=False
            )
        start_epoch = checkpoint_epoch + 1
    else:
        gs = 0
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(cfg)

    # Perform the training loop.
    logger.info("Start epoch: {} gs {}".format(start_epoch + 1, gs+1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            if cfg.TRAIN.USE_CENTER_VALIDATION:
                validation_epoch_center(val_loader, model, val_meter, cur_epoch, cfg)
            else:
                validation_epoch(val_loader, model, val_meter, cur_epoch, cfg)
        # Train for one epoch.
        gs = train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, gs, cfg)

        # Compute precise BN stats.
        # if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
        #     calculate_and_update_precise_bn(
        #         train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
        #     )
        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, gs, cfg)