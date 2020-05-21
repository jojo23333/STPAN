import torch

def rebase_flow_2d(flow_2d, kernel_s, MAX_OFFSET_S=128):
    """
    flow_2d: [B, K, 2, H, W]
    """
    mid = kernel_s // 2
    for i in range(kernel_s):
        for j in range(kernel_s):
            sid = i * kernel_s + j
            flow_2d[:, sid, 0, :, :] = flow_2d[:, sid, 0, :, :] * MAX_OFFSET_S + (j-mid)
            flow_2d[:, sid, 0, :, :] = flow_2d[:, sid, 1, :, :] * MAX_OFFSET_S + (i-mid)
    return flow_2d

def rebase_flow_3d(flow_3d, kernel_s, kernel_t, FRAME_SIZE=5, MAX_OFFSET_S=128):
    mid = kernel_s // 2
    mid_t = kernel_t//2
    for t in range(kernel_t):
        for i in range(kernel_s):
            for j in range(kernel_s):
                sid = t * kernel_s * kernel_s + i * kernel_s + j
                flow_3d[:, sid, 0, :, :] = flow_3d[:, sid, 0, :, :] * MAX_OFFSET_S + (j-mid)
                flow_3d[:, sid, 1, :, :] = flow_2d[:, sid, 1, :, :] * MAX_OFFSET_S + (i-mid)
                flow_3d[:, sid, 2, :, :] = flow_2d[:, sid, 2, :, :] * (FRAME_SIZE // 2) + (t-mid_t)
    return flow_3d

def spatial_sampler(input_frames, flow_2d):
    """
    This function perform as spatio-temporal sampler
    
    Input:
    input_frames: [B, C, H, W]
    3D_flow: [B, sample_num, 2, H, W]
    t_origin: predicted frame id 

    Output:
    generated img yt_
    [B, C, H, W]
    """
    B, C, H, W = input_frames.size()
    K = flow_3d.size()[1]
    input_frames = input_frames.view(B, -1, C, H, W)

    ind_b = torch.arange(0, B)
    ind_h = torch.arange(0, H)
    ind_w = torch.arange(0, W)
    ind_k = torch.zeros([K])
    ind_b, ind_k, ind_h, ind_w = torch.meshgrid(ind_b, ind_t, ind_h, ind_w)

    b = ind_b
    k = ind_k
    x = ind_w + flow_3d[:,:,0,:,:]
    y = ind_h + flow_3d[:,:,1,:,:]

    x0 = torch.floor(x).float()
    x1 = x0 + 1
    y0 = torch.floor(y).float()
    y1 = y0 + 1

    x0 = x0.clamp(0, W-1)
    x1 = x1.clamp(0, W-1)
    y0 = y0.clamp(0, H-1)
    y1 = y1.clamp(0, H-1)

    w0 = (x - x0)*(y - y0)
    w1 = (x1 - x)*(y - y0)
    w2 = (x - x0)*(y1 - y)
    w3 = (x1 - x)*(y1 - y)

    I0 = input_frames[b, k, :, y1, x1]
    I1 = input_frames[b, k, :, y1, x0]
    I2 = input_frames[b, k, :, y0, x1]
    I3 = input_frames[b, k, :, y0, x0]
    out = w0*I0 + w1*I1 + w2*I2 + w3*I3
    out = out.permute([0, 1, 4, 2, 3])
    return out



def spatial_temporal_sampler(input_frames, flow_3d, t_origin):
    """
    This function perform as spatio-temporal sampler
    
    Input:
    input_frames: [B, T, C, H, W]
    3D_flow: [B, sample_num, 3, H, W]
    t_origin: predicted frame id 

    Output:
    generated img yt_
    [B, C, H, W]
    """
    B, T, C, H, W = input_frames.size()
    K = flow_3d.size()[1]

    ind_b = torch.arange(0, B)
    ind_h = torch.arange(0, H)
    ind_w = torch.arange(0, W)
    ind_t = torch.ones([K]) * t_origin
    ind_b, ind_t, ind_h, ind_w = torch.meshgrid(ind_b, ind_t, ind_h, ind_w)

    b = ind_b
    x = ind_w + flow_3d[:,:,0,:,:]
    y = ind_h + flow_3d[:,:,1,:,:]
    t = ind_t + flow_3d[:,:,2,:,:]

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    t0 = torch.floor(t)
    t1 = t0 + 1

    x0 = x0.clamp(0, W-1)
    x1 = x1.clamp(0, W-1)
    y0 = y0.clamp(0, H-1)
    y1 = y1.clamp(0, H-1)
    t0 = t0.clamp(0, T-1)
    t1 = t1.clamp(0, T-1)

    w0 = (x1-x) * (y1-y) * (t1-t)
    w1 = (x1-x) * (y1-y) * (t-t0)
    w2 = (x1-x) * (y-y0) * (t1-t)
    w3 = (x1-x) * (y-y0) * (t-t0)
    w4 = (x-x0) * (y1-y) * (t1-t)
    w5 = (x-x0) * (y1-y) * (t-t0)
    w6 = (x-x0) * (y-y0) * (t1-t)
    w7 = (x-x0) * (y-y0) * (t-t0)

    I0 = input_frames[b, t0, :, y0, x0]
    I1 = input_frames[b, t1, :, y0, x0]
    I2 = input_frames[b, t0, :, y1, x0]
    I3 = input_frames[b, t1, :, y1, x0]
    I4 = input_frames[b, t0, :, y0, x1]
    I5 = input_frames[b, t1, :, y0, x1]
    I6 = input_frames[b, t0, :, y1, x1]
    I7 = input_frames[b, t1, :, y1, x1]

    out = w0*I0 + w1*I1 + w2*I2 + w3*I3 + w4*I4 + w5*I5 + w6*I6 + w7*I7
    out = out.permute([0, 1, 4, 2, 3])
    return out