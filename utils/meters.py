import datetime
import torch
import os
import cv2
import utils.logging as logging
import utils.misc as misc
import numpy as np

from collections import deque
from fvcore.common.timer import Timer
from os.path import join as Join

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measures training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # current psnr 
        self.psnr = ScalarMeter(cfg.LOG_PERIOD)
        self.psnr_total = 0.0
        # base psnr
        self.base_psnr = ScalarMeter(cfg.LOG_PERIOD)
        self.base_psnr_total = 0.0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.psnr.reset()
        self.psnr_total = 0.0
        self.base_psnr.reset()
        self.base_psnr_total = 0.0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, loss, psnr, base_psnr, lr, mb_size):
        """
        Update the current stats.
        Args:
            psnr (float): psnr
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.loss.add_value(loss)
        self.psnr.add_value(psnr)
        self.base_psnr.add_value(base_psnr)
        self.lr = lr
        # Aggregate stats
        self.loss_total += loss * mb_size
        self.psnr_total += psnr * mb_size
        self.base_psnr_total += base_psnr * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "psnr": self.psnr.get_win_median(),
            "base_psnr": self.base_psnr.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        avg_loss = self.loss_total / self.num_samples
        avg_psnr = self.psnr_total / self.num_samples
        avg_base_psnr = self.base_psnr_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "psnr": avg_psnr,
            "base_psnr": avg_base_psnr,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

class TestMeter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.forward_timer = Timer()
        self.total_time = 0
        self.cnt = 0
        self.score = dict()
        self.output_dir = Join(cfg.TEST.OUTPUT_DIR, cfg.TEST.DATASET)
        self.save_img = cfg.TEST.SAVE_IMG
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.score_csv = open(Join(self.output_dir, "score.csv"), 'w')
        self.score_csv.write("vid, image_id, psnr, ssim\n")

    def forward_tic(self):
        """
        Start to record time.
        """
        self.forward_timer.reset()

    def forward_toc(self):
        """
        Stop to record time.
        """
        self.forward_timer.pause()
        self.total_time += self.forward_timer.seconds()
        self.cnt += 1
    
    def log_img_result(self, img_out, vid, img_id, psnr, ssim):
        if vid not in self.score.keys():
            self.score[vid] = {}
        
        # log score
        self.score[vid][img_id] = (psnr, ssim)
        self.score_csv.write("{},{},{},{}\n".format(vid, img_id, psnr, ssim))

        # save img
        if self.save_img:
            # if not os.path.exists(Join(self.output_dir, vid)):
            #     os.makedirs(Join(self.output_dir, vid))
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(Join(self.output_dir, img_id), img_out)
    
    def log_average_score(self):
        score_per_vid = {}
        for vid in self.score.keys():
            psnrs = [x[0] for x in self.score[vid].values()]
            ssims = [x[1] for x in self.score[vid].values()]
            score_per_vid[vid] = (np.mean(psnrs), np.mean(ssims))
        
        with open(Join(self.output_dir, 'videos_scores.csv'), 'w') as f:
            f.write('video_id, psnr, ssim\n')
            for vid in self.score.keys():
                f.write("{},{},{}\n".format(vid, score_per_vid[vid][0], score_per_vid[vid][1]))
        return score_per_vid

    def speed(self):
        return self.total_time, self.total_time / self.cnt


class ValMeter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.score = dict()

    def reset(self):
        del self.score
        self.score = dict()
    
    def log_img_result(self, vid, img_id, psnr, ssim):
        if vid not in self.score.keys():
            self.score[vid] = {}
        
        # log score
        self.score[vid][img_id] = (psnr, ssim)
        # self.score_csv.write("{},{},{},{}\n".format(vid, img_id, psnr, ssim))

        # # save img
        # if not os.path.exists(Join(self.output_dir, vid)):
        #     os.makedirs(Join(self.output_dir, vid))
        # img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR) 
        # cv2.imwrite(img_out, Join(self.output_dir, vid, img_id + '.png'))
    
    def log_average_score(self, cur_epoch):
        score_per_vid = {}
        for vid in self.score.keys():
            psnrs = [x[0] for x in self.score[vid].values()]
            ssims = [x[1] for x in self.score[vid].values()]
            score_per_vid[vid] = (np.mean(psnrs), np.mean(ssims))
        avg_psnr = np.mean([v[0] for k, v in score_per_vid.items()])
        avg_ssim = np.mean([v[1] for k, v in score_per_vid.items()])

        stats = {
            "cur_epoch": cur_epoch,
            "_type": "validation_epoch",
            "psnr": avg_psnr,
            "ssim": avg_ssim
        }
        logging.log_json_stats(stats)
        logging.log_json_stats(score_per_vid)
        # with open(Join(self.output_dir, 'videos_scores.csv'), 'w') as f:
        #     f.write('video_id, psnr, ssim\n')
        #     for vid in self.score.keys():
        #         f.write("{},{},{}\n".format(vid, score_per_vid[vid][0], score_per_vid[vid][1]))
        return 
