import torch.nn as nn
import torch
from models.arch.dct import H264Transform
from torch.nn.modules.loss import _Loss

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        dx = x[:,:,:,1:] - x[:,:,:,:-1]
        dy = x[:,:,1:,:] - x[:,:,:-1,:]
        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return (torch.mean(grad_diff_x ** alpha) + torch.mean(grad_diff_y ** alpha)) / 2.

class MseGradientLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', lambda1=1., lambda2=1.):
        super(MseGradientLoss, self).__init__(size_average, reduce, reduction)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def forward(self, inputs, gt):
        g_loss = gradient_loss(inputs, gt)
        mse_loss = nn.functional.mse_loss(inputs, gt, reduction=self.reduction)
        return self.lambda1 * g_loss + self.lambda2 * mse_loss


class AnnealingLoss(_Loss)
    def __init__(self, size_average=None, reduce=None, reduction='mean', anneal_alpha=1.):
        super(AnnealingLoss, self).__init__(size_average, reduce, reduction)
        self.anneal_alpha = anneal_alpha

    def forward(self, weighted_samples, gt, gs):
        anneal_rate = torch.pow(self.anneal_alpha, gs) * 100
        



_LOSSES = {"l1": nn.L1Loss,
            "l2": nn.MSELoss,
            "l2_with_gradient": MseGradientLoss
           }

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr