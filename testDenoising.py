import torch
import numpy as np
import pprint

import models.model_builder as model_builder
import utils.distributed as du
import utils.checkpoint as cu
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import datasets.loader as loader

from tqdm import tqdm
from utils.meters import TestMeter
from utils.image import tensor2uint, split_test, merge_test

logger = logging.get_logger(__name__)

@torch.no_grad()
def evaluate(test_loader, model, test_meter, cfg):
    """
        Evaluate model
    """
    model.eval()

    # TODO test meter setting

    for inputs, gt, frame_ids in tqdm(test_loader):
        batch = split_test(inputs, cfg.TEST.TEST_PATCH_SIZE)
        preds = []
        for ip in batch:
            test_meter.forward_tic()
            preds.append(model(ip.contiguous().cuda(non_blocking=True)).cpu())
            test_meter.forward_toc()
        pred = merge_test(preds, inputs[:,0,:,:,:], cfg.TEST.TEST_PATCH_SIZE)

        for i, fid in enumerate(frame_ids):
            vid, img_id = frame_ids[i].split('/')
            print(vid, img_id)

            pred = tensor2uint(pred[i, ...])
            psnr, ssim = -1, -1
            if gt[0] != -1:
                gt = tensor2uint(gt[i, ...])
                ssim = metrics.calculate_ssim(pred, gt)
                psnr = metrics.calculate_psnr(pred, gt)

            # save img
            # save statistic of each img
            test_meter.log_img_result(pred, vid, img_id, psnr, ssim) 

    test_meter.log_average_score()

    return


@torch.no_grad()
def evaluate_with_augmentation(test_loader, model, test_meter, cfg):
    """
        Evaluate model
    """
    model.eval()

    # TODO test meter setting

    for inputs, gt, frame_ids in tqdm(test_loader):
        
        preds_aug = []
        ip0 = inputs.flip(-1)
        batch = split_test(ip0, cfg.TEST.TEST_PATCH_SIZE)
        preds = []
        for ip in batch:
            test_meter.forward_tic()
            preds.append(model(ip.contiguous().cuda(non_blocking=True)).cpu())
            test_meter.forward_toc()
        pred = merge_test(preds, ip0[:,0,:,:,:], cfg.TEST.TEST_PATCH_SIZE)
        preds_aug.append(pred.flip(-1))

        ip0 = inputs.flip(-2)
        batch = split_test(ip0, cfg.TEST.TEST_PATCH_SIZE)
        preds = []
        for ip in batch:
            test_meter.forward_tic()
            preds.append(model(ip.contiguous().cuda(non_blocking=True)).cpu())
            test_meter.forward_toc()
        pred = merge_test(preds, ip0[:,0,:,:,:], cfg.TEST.TEST_PATCH_SIZE)
        preds_aug.append(pred.flip(-2))

        ip0 = inputs.transpose(-1, -2)
        batch = split_test(ip0, cfg.TEST.TEST_PATCH_SIZE)
        preds = []
        for ip in batch:
            test_meter.forward_tic()
            preds.append(model(ip.contiguous().cuda(non_blocking=True)).cpu())
            test_meter.forward_toc()
        pred = merge_test(preds, ip0[:,0,:,:,:], cfg.TEST.TEST_PATCH_SIZE)
        preds_aug.append(pred.transpose(-1, -2))

        ip0 = inputs.transpose(-1, -2).flip(-2)
        batch = split_test(ip0, cfg.TEST.TEST_PATCH_SIZE)
        preds = []
        for ip in batch:
            test_meter.forward_tic()
            preds.append(model(ip.contiguous().cuda(non_blocking=True)).cpu())
            test_meter.forward_toc()
        pred = merge_test(preds, ip0[:,0,:,:,:], cfg.TEST.TEST_PATCH_SIZE)
        preds_aug.append(pred.flip(-2).transpose(-1, -2))

        ip0 = inputs.transpose(-1, -2).flip(-1)
        batch = split_test(ip0, cfg.TEST.TEST_PATCH_SIZE)
        preds = []
        for ip in batch:
            test_meter.forward_tic()
            preds.append(model(ip.contiguous().cuda(non_blocking=True)).cpu())
            test_meter.forward_toc()
        pred = merge_test(preds, ip0[:,0,:,:,:], cfg.TEST.TEST_PATCH_SIZE)
        preds_aug.append(pred.flip(-1).transpose(-1, -2))

        ip0 = inputs.flip(-1).flip(-2)
        batch = split_test(ip0, cfg.TEST.TEST_PATCH_SIZE)
        preds = []
        for ip in batch:
            test_meter.forward_tic()
            preds.append(model(ip.contiguous().cuda(non_blocking=True)).cpu())
            test_meter.forward_toc()
        pred = merge_test(preds, ip0[:,0,:,:,:], cfg.TEST.TEST_PATCH_SIZE)
        preds_aug.append(pred.flip(-2).flip(-1))

        pred = torch.mean(torch.stack(preds_aug, dim=0), dim=0)

        for i, fid in enumerate(frame_ids):
            vid, img_id = frame_ids[i].split('/')
            print(vid, img_id)

            pred = tensor2uint(pred[i, ...])
            psnr, ssim = -1, -1
            if gt[0] != -1:
                gt = tensor2uint(gt[i, ...])
                ssim = metrics.calculate_ssim(pred, gt)
                psnr = metrics.calculate_psnr(pred, gt)

            # save img
            # save statistic of each img
            test_meter.log_img_result(pred, vid, img_id, psnr, ssim) 

    test_meter.log_average_score()

    return


        

def test(cfg):
    """
    Test a model
    """
    logging.setup_logging(logger, cfg)

    logger.info("Test with config")
    logger.info(pprint.pformat(cfg))

    model = model_builder.build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model)

    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        gs, checkpoint_epoch = cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer=None,
            inflation=False,
            convert_from_caffe2=False
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        gs, checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, None
        )
        start_epoch = checkpoint_epoch + 1

    # Create the video train and val loaders.
    test_loader = loader.construct_loader(cfg, "test")

    test_meter = TestMeter(cfg)

    if cfg.TEST.AUGMENT_TEST:
        evaluate_with_augmentation(test_loader, model, test_meter, cfg)
    else:
        evaluate(test_loader, model, test_meter, cfg)



