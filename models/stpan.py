import torch
import torch.nn as nn
import torch.nn.functional as F
from models.arch.samplers import rebase_flow_3d
from models.arch.samplers import spatial_temporal_sampler
from utils.image import sRGBforward

def unet_subset(ch, N, D=3, norm_type="none"):
    """
    constructing Encoder & Decoder layers recursively
    inputs:
        inputs: input feature map
        N: the number of time of downsampling
        D: number of continious conv layers in with each img size
        ch: the output channel number of current level of layers
        norm_type: DEPRECATED. use default none.
    output:
        inputs: output feature map
    """
    print("Average pool down sample")
    inputs = tf.layers.average_pooling2d(inputs, pool_size=2, strides=2)
    for i in range(D):
        print('Pre-Layer with {} channels at N={}'.format(ch, N))
        inputs = conv2d(inputs, ch, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
    
    if N > 0:
        ch_next = ch * 2 if ch < 512 else ch
        inputs = unet_subset(inputs, ch_next, N-1, D, norm_type=norm_type)
        for i in range(D):
            print('Post-Layer with {} channels at N={}'.format(ch, N))
            inputs = conv2d(inputs, ch, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)

    print("Bilinear up sample")
    inputs = upsample_layer(inputs, method=tf.image.ResizeMethod.BILINEAR)
    inputs = tf.concat([inputs, _inputs], axis=-1)
    return inputs

def make_conv_blocks(in_ch, ch, layers):
    conv_blocks = []
    conv_blocks.append(nn.Conv2d(in_ch, ch, 3, 1, 1))
    conv_blocks.append(nn.ReLU())
    for i in range(layers - 1):
        conv_blocks.append(nn.Conv2d(ch, ch, 3, 1, 1))
        conv_blocks.append(nn.ReLU())
    return nn.Sequential(conv_blocks)

class UnetSubset(nn.Module):
    def __init__(self, in_ch, cfg):
        super(UnetSubset, self).__init__()
        self.cfg = cfg
        self.fea_ch = [64, 128, 256, 512, 512]

        self.average_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.l1_down = make_conv_blocks(in_ch, self.fea_ch[1], 3)
        self.l2_down = make_conv_blocks(self.fea_ch[2], self.fea_ch[2], 3)
        self.l3_down = make_conv_blocks(self.fea_ch[3], self.fea_ch[3], 3)
        self.l4 = make_conv_blocks(self.fea_ch[4], self.fea_ch[4], 3)

        self.l3_up = make_conv_blocks(self.fea_ch[3]*2, self.fea_ch[3], 3)
        self.l2_up = make_conv_blocks(self.fea_ch[2]*2, self.fea_ch[2], 3) 
        self.l1_conv = nn.Conv2d(self.fea_ch[2]*2, self.fea_ch[2])

    def forward(self, fea):
        l1_fea = self.average_pooling(fea)
        l1_fea = self.l1_down(l1_fea)
        
        l2_fea = self.average_pooling(l1_fea)
        l2_fea = self.l2_down(l2_fea)

        l3_fea = self.average_pooling(l2_fea)
        l3_fea = self.l3_down(l3_fea)

        l4_fea = self.average_pooling(l3_fea)
        l4_fea = self.l4(l4_fea)

        l3_fea_up = F.interpolate(l4_fea, scale_factor=2, mode='bilinear', align_corners=False)
        l3_fea_up = self.l3_up(torch.cat([l3_fea_up, l3_fea], dim=1))
        
        l2_fea_up = F.interpolate(l3_fea_up, scale_factor=2, mode='bilinear', align_corners=False)
        l2_fea_up = self.l1_up(torch.cat([l2_fea_up, l2_fea], dim=1))

        l1_fea_up = F.interpolate(l2_fea_up, scale_factor=2, mode='bilinear', align_corners=False)
        l1_fea = self.relu(self.l1_conv(torch.cat([l1_fea_up, l1_fea], dim=1)))
        return l1_fea


class STPAN(nn.Module):
    def __init__(self, cfg):
        super(ST_PAN, self).__init__()
        KERNEL_K = cfg.STPAN.KERNEL_K
        KERNEL_T = cfg.STPAN.KERNEL_T
        FRAME_SIZE = cfg.DENOISE.NUM_FRAMES
        COLOR_CH = cfg.STPAN.COLOR_CH
        self.kernel_k = KERNEL_K
        self.kernel_t = KERNEL_T
        self.frame_size = FRAME_SIZE
        self.color_ch = COLOR_CH
        self.anneal_alpha = cfg.STPAN.ANNEAL_ALPHA
        self.noise_type = cfg.NOISE_TYPE

        self.num_ch = 128
        self.ch_flow = KERNEL_K * KERNEL_K * KERNEL_T
        self.conv_first_1 = make_conv_blocks(3, 64, 3)
        self.unet_subset = unet_subset(in_ch=64, cfg)

        self.flow_recon_1 = make_conv_blocks(self.num_ch, self.num_ch, 2)

        self.flow_recon_2 = nn.Sequential([
            nn.Conv2d(self.num_ch + FRAME_SIZE * COLOR_CH, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, self.ch_flow * 3, 3, 1, 1),
        ])
        
        self.weight_recon_1 = make_conv_blocks(self.num_ch, self.num_ch, 2)

        self.weight_recon_2 = nn.Sequential([
            nn.Conv2d(self.num_ch + KERNEL_K * KERNEL_K * KERNEL_T * COLOR_CH + FRAME_SIZE * COLOR_CH, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, self.ch_flow, 3, 1, 1),
        ])
        
        if self.noise_type == 'gaussian'
            srgb_transfer_func = sRGBforward
        else:
            srgb_transfer_func = lambda x: x

    def forward(self, inputs_dict, gs):
        input_frames = inputs_dict["inputs"]
        gt = inputs_dict["gt"]
        noise_level = inputs_dict["noise_level"]
        B, T, C, H, W = input_frames.size()

        if self.noise_type == 'gaussian':
            inputs = input_frames.view(B, -1, H, W)
        elif self.noise_type == 'read_and_shot':
            sig_read, sig_shot = noise_level
            sig_read = torch.FloatTensor(sig_read).view(B, 1, 1, 1)
            sig_shot = torch.FloatTensor(sig_shot).view(B, 1, 1, 1)
            noise_level = torch.sqrt(sig_read ** 2 + sig_shot ** 2 * noisy_frames[:, self.frame_size // 2, ...])
            inputs = torch.stack([input_frames.view(B, -1, H, W), noise_level], dim=1)

        fea = self.conv_first_1(inputs)
        fea = self.unet_subset(fea)

        flow_weights_fea = fea[:, :128, ...]
        flow_fea = fea[:, 128:, ...]

        flow_fea = F.interpolate(self.flow_recon_1(flow_fea), scale_factor=2, mode='bilinear', align_corners=False)
        flow_3d = self.flow_recon_2(torch.cat([x, flow_fea], dim=1))
        flow_3d = flow_3d.view(B, self.ch_flow, 3, h, w)
        flow_3d = rebase_flow_3d(flow_3d, self.kernel_s, self.kernel_t)
        sample_points = spatial_temporal_sampler(x, flow_3d, T//2)
        
        weights = self.weight_recon_1(flow_weights_fea)
        weights = self.flow_recon_2(torch.cat([weights, 
                                               sample_points.view(B, -1, H, W), 
                                               x.view(B, -1, H, W)]), dim=1)
        weights = weights.view(B, self.ch_flow, 1, H, W)
        out = weights * sample_points
        pred = out.sum(dim=1)

        if self.training:
            K = self.kernel_k * self.kernel_k
            annealing_loss = 0
            annealing_rate = torch.pow(self.anneal_alpha, gs) * 100
            for i in range(self.kernel_t):
                part_sum = out[:, i*K:i*K+K, ...].sum(dim=1)
                annealing_loss += F.mse_loss(srgb_transfer_func(part_sum), gt)
            mse = F.mse_loss(srgb_transfer_func(pred), gt)
            losses = {"mse_loss": mse, "annealing_loss": annealing_loss * annealing_rate}
            
            psnr =  20 * torch.log10(1. / torch.sqrt(mse))
            base_mse = torch.mean((srgb_transfer_func(input_frames[:, self.frame_size // 2, ...]) - gt) ** 2)
            base_psnr = 20 * torch.log10(1. / torch.sqrt(base_mse))
            training_info = {'annealing rate': annealing_rate, 'psnr': psnr, 'base psnr': base_psnr}

            return losses, training_info
        else:
            return pred



        



        