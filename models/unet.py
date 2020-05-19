import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        return identity + out


class DenseBlock_noBN(nn.Module):
    '''Denseblock without BN
    '''
    def __init__(self, nf=64, num_layers=5):
        super(DenseBlock_noBN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nf*3, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nf*4, nf, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, x):
        f1 = self.relu(self.conv1(x))
        f2 = self.relu(self.conv2(f1))
        f3 = self.relu(self.conv3(torch.cat([f1, f2], 1)))
        f4 = self.relu(self.conv4(torch.cat([f1, f2, f3], 1)))
        f5 = self.relu(self.conv5(torch.cat([f1, f2, f3, f4], 1)))
        return f5 + x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )
        initialize_weights(self.up)

    def forward(self, x):
        x = self.up(x)
        return x


class PointWiseGlobalFusion(torch.nn.Module):
    def __init__(self, nframes, in_ch, nf=64, nl_groups=4):
        super(PointWiseGlobalFusion, self).__init__()
        self.nframes = nframes
        self.nl_groups = nl_groups
        self.tAtt_1 = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=True, groups=nl_groups)
        self.tAtt_2 = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=True, groups=nl_groups)
        self.max_pooling = nn.MaxPool3d((nframes, 1, 1), stride=1, padding=0)

    def forward(self, x):
        B, N, C, H, W = x.size()
        assert N == self.nframes

        ref = self.nframes // 2
        # non local similarities
        emb1 = self.tAtt_1(x.view(-1, C, H, W)).view(B, N, -1, H, W)        
        emb2 = self.tAtt_2(x.view(-1, C, H, W)).view(B, N, -1, H, W)
        emb1 = emb1.view(B, N, 1, self.nl_groups, -1, H, W)
        emb2 = emb2.view(B, 1, N, self.nl_groups, -1, H, W)
        
        cor_weight = torch.sum(emb1 * emb2, 4, keepdim=True)
        cor_weight = F.softmax(cor_weight, dim=2)
        nl_feat = cor_weight * x.view(B, 1, N, self.nl_groups, -1, H, W)
        nl_feat = torch.sum(nl_feat, 2).view(B, N, -1, H, W) # [B, N, nl_grooups, c', H, W]

        # Global pooling features
        pooling_feat = self.max_pooling(x.permute(0, 2, 1, 3, 4)).view(B, 1, -1, H, W)
        pooling_feat = pooling_feat.repeat(1, N, 1, 1, 1) # [B, N, C, H, W]

        global_feat = torch.cat([x, nl_feat, pooling_feat], dim=2)
        return global_feat


class Unet(torch.nn.Module):
    def __init__(self, cfg, block=ResidualBlock_noBN, in_ch=3, out_ch=3):
        super(Unet, self).__init__()
        self.NF = cfg.UNET.NUM_FILTERS 
        self.nframes = cfg.DATA.NUM_FRAMES
        # self.CH_UP = cfg.UNET.CHANNEL_DECODER
        
        self.up_sampler = up_conv
        #functools.partial(F.interpolate, scale_factor=2, mode='bilinear', align_corners=False)
        
        basicblock = block

        self.conv_first = nn.Conv2d(in_ch, self.NF[0], 3, 1, 1, bias=True)

        self.L1_1 = basicblock(nf=self.NF[0])
        self.L1_D = nn.Conv2d(self.NF[0], self.NF[1], 3, 2, 1, bias=True)
        self.L2_1 = basicblock(nf=self.NF[1])
        self.L2_D = nn.Conv2d(self.NF[1], self.NF[2], 3, 2, 1, bias=True)
        self.L3_1 = basicblock(nf=self.NF[2])
        self.L3_D = nn.Conv2d(self.NF[2], self.NF[3], 3, 2, 1, bias=True)
        self.L4 = ResidualBlock_noBN(nf=self.NF[3])

        self.L3_U = self.up_sampler(in_ch=self.NF[3], out_ch=self.NF[2])
        self.L3_2 = nn.Sequential(nn.Conv2d(self.NF[2]*2, self.NF[2], 1, 1, 0), 
                                    basicblock(nf=self.NF[2]))
        self.L2_U = self.up_sampler(in_ch=self.NF[2], out_ch=self.NF[1])
        self.L2_2 = nn.Sequential(nn.Conv2d(self.NF[1]*2, self.NF[1], 1, 1, 0), 
                                    basicblock(nf=self.NF[1]))
        self.L1_U = self.up_sampler(in_ch=self.NF[1], out_ch=self.NF[0])
        self.L1_2 = nn.Sequential(nn.Conv2d(self.NF[0]*2, self.NF[0], 1, 1, 0), 
                                    basicblock(nf=self.NF[0]))

        self.last_conv = nn.Conv2d(self.NF[0], out_ch, 1)

    def forward(self, x):
        B, N, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        assert N == self.nframes

        x_1 = self.L1_1(self.conv_first(x))

        x_2 = self.L2_1(self.L1_D(x_1))
        x_3 = self.L3_1(self.L2_D(x_2))
        x_4 = self.L4(self.L3_D(x_3))

        x_3u = torch.cat((self.L3_U(x_4), x_3), dim=1)
        x_3 = self.L3_2(x_3u)
        x_2u = torch.cat((self.L2_U(x_3), x_2), dim=1)
        x_2 = self.L2_2(x_2u)
        x_1u = torch.cat((self.L1_U(x_2), x_1), dim=1)
        x_1 = self.L1_2(x_1u)

        out = self.last_conv(x_1)

        return out

class GlobalFusionUnet(Unet):
    def __init__(self, cfg, block=ResidualBlock_noBN, in_ch=3, out_ch=3):
        super(GlobalFusionUnet, self).__init__(cfg, block, in_ch, out_ch)
        NF = cfg.UNET.NUM_FILTERS 
        nl_blocks = cfg.UNET.NUM_FUSION_NL_GROUPS
        nframes = cfg.DATA.NUM_FRAMES

        self.L1_1_Fusion = PointWiseGlobalFusion(nframes, in_ch=NF[0], nf=NF[0], nl_groups=nl_blocks)
        self.L1_1_1x1 = nn.Conv2d(NF[0]*3, NF[0], 1, 1, 0)

        self.L2_1_Fusion = PointWiseGlobalFusion(nframes, in_ch=NF[1], nf=NF[0], nl_groups=nl_blocks)
        self.L2_1_1x1 = nn.Conv2d(NF[1]*3, NF[1], 1, 1, 0)
        
        self.L3_1_Fusion = PointWiseGlobalFusion(nframes, in_ch=NF[2], nf=NF[0], nl_groups=nl_blocks)
        self.L3_1_1x1 = nn.Conv2d(NF[2]*3, NF[2], 1, 1, 0)

        self.L1_2_Fusion = PointWiseGlobalFusion(nframes, in_ch=NF[0], nf=NF[0], nl_groups=nl_blocks)
        self.L1_2_1x1 = nn.Conv2d(NF[0]*4, NF[0]*2, 1, 1, 0)
       
        self.L2_2_Fusion = PointWiseGlobalFusion(nframes, in_ch=NF[1], nf=NF[0], nl_groups=nl_blocks)
        self.L2_2_1x1 = nn.Conv2d(NF[1]*4, NF[1]*2, 1, 1, 0)

        self.L3_2_Fusion = PointWiseGlobalFusion(nframes, in_ch=NF[2], nf=NF[0], nl_groups=nl_blocks)
        self.L3_2_1x1 = nn.Conv2d(NF[2]*4, NF[2]*2, 1, 1, 0)

    def forward(self, x):
        B, N, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        assert N == self.nframes

        x_1 = self.L1_1(self.conv_first(x))
        x_1 = self.L1_1_Fusion(x_1.view(B, N, -1, H, W)).view(B*N, -1, H, W)
        x_1 = self.L1_1_1x1(x_1)

        x_2 = self.L2_1(self.L1_D(x_1))
        x_2 = self.L2_1_Fusion(x_2.view(B, N, -1, H//2, W//2)).view(B*N, -1, H//2, W//2)
        x_2 = self.L2_1_1x1(x_2)

        x_3 = self.L3_1(self.L2_D(x_2))
        x_3 = self.L3_1_Fusion(x_3.view(B, N, -1, H//4, W//4)).view(B*N, -1, H//4, W//4)
        x_3 = self.L3_1_1x1(x_3)

        x_4 = self.L4(self.L3_D(x_3))

        x_3u = self.L3_U(x_4)
        x_3u = self.L3_2_Fusion(x_3u.view(B, N, -1, H//4, W//4)).view(B*N, -1, H//4, W//4)
        x_3u = self.L3_2_1x1(torch.cat((x_3u, x_3), dim=1))
        x_3  = self.L3_2(x_3u)

        x_2u = self.L2_U(x_3)
        x_2u = self.L2_2_Fusion(x_2u.view(B, N, -1, H//2, W//2)).view(B*N, -1, H//2, W//2)
        x_2u = self.L2_2_1x1(torch.cat((x_2u, x_2), dim=1))
        x_2  = self.L2_2(x_2u)

        x_1u = self.L1_U(x_2)
        x_1u = self.L1_2_Fusion(x_1u.view(B, N, -1, H, W)).view(B*N, -1, H, W)
        x_1u = self.L1_2_1x1(torch.cat((x_1u, x_1), dim=1))
        x_1  = self.L1_2(x_1u)

        out = self.last_conv(x_1)

        return out


class VideoUnet(nn.Module):

    def __init__(self, cfg):
        super(VideoUnet, self).__init__()
        self.feat_ch = cfg.UNET.NUM_FILTERS[0]
        self.num_frames = cfg.DATA.NUM_FRAMES
        if cfg.UNET.USE_DENSE_BLOCK:
            block = DenseBlock_noBN
        else:
            block = ResidualBlock_noBN

        if cfg.UNET.USE_GLOBAL_FUSION:
            self.backbone = GlobalFusionUnet(cfg, block, in_ch=3, out_ch=self.feat_ch)
        else:
            self.backbone = Unet(cfg, block, in_ch=3, out_ch=self.feat_ch)

        self.center = self.num_frames // 2
        self.fusion = nn.Sequential(
            nn.Conv2d(self.feat_ch * self.num_frames, self.feat_ch, 1, 1),
            block(nf=self.feat_ch),
            block(nf=self.feat_ch),
            block(nf=self.feat_ch)
        )
        self.last_conv = nn.Conv2d(self.feat_ch, 3, 3, 1, 1, bias=True)
    
    def forward(self, x):
        B, N, C, H, W = x.size()
        x_center = x[:, self.center, :, :, :] 

        feat = self.backbone(x)
        feat = feat.view(B, -1, H, W)
        feat = self.fusion(feat)
        
        res_x = self.last_conv(feat)

        return res_x + x_center
        # return res_x

        