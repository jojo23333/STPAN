import datetime
import torch
import os
import cv2
import utils.logging as logging
import utils.misc as misc
import utils.distributed as du
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
        self.log_period = cfg.LOG_PERIOD

        self.infos = None
        self.num_samples = 0

    def init(self, keys):
        self.infos = {}
        for key in keys:
            self.infos[key] = ScalarMeter(self.log_period)

    def reset(self):
        """
        Reset the Meter.
        """
        for k, v in self.infos.items():
            v.reset()

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

    def update_stats(self, info_dict):
        """
        Update the current stats.
        Args:
            psnr (float): psnr
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        if self.infos is None:
            self.init(info_dict.keys())
        # reduce from all gpus
        if self._cfg.NUM_GPUS > 1:
            for k, v in info_dict.items():
                info_dict[k] = du.all_reduce([v])
        # syncronize from gpu to cpu
        info_dict = {k: v.item() for k, v in info_dict.items()}
        # log value into scalar meter
        for k, v in info_dict.items():
            self.infos[k].add_value(v)

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
            "mem": int(np.ceil(mem_usage)),
        }
        infos = {k: v.get_win_avg() for k, v in self.infos}
        stats.update(infos)
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
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "mem": int(np.ceil(mem_usage)),
        }
        infos = {k: v.get_global_avg() for k, v in self.infos}
        stats.update(infos)
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
