import tensorflow as tf
import numpy as np    

from train_tools import per_pixel_convolve
from arch import spatial_temporal_sampler, none_local_blocks_spatial, conv2d, upsample_layer
FLAGS = tf.flags.FLAGS

def stpan(input_frames, noise_level, noise_type, C=3, anneal_term=1, norm_type="none", batch_size=8):
    """
    input:  input_frames: [B, H, W, T, C]
            noise_level: noise level
            noise_type: noise type
            C: channel number
            anneal_term: desgined for flow anneal. DEPRECATED.
            norm type: insert normalization layer after each later. DEPRECATED.
            batch_size: batch size
    output: 
            out: clean image
            op_flow: predicted kernel offset
            weights: predicted kernel weights
            per_points_out: filtered per sample point image
    """
    print(input_frames.get_shape())
    KERNEL_K = FLAGS.kernel_size_k
    KERNEL_T = FLAGS.kernel_size_t
    FRAME_SIZE = FLAGS.frame_size
    with tf.name_scope("stpan"):
        B, H, W, T, CH = input_frames.get_shape().as_list()
        origin = T // 2
        assert(T == FRAME_SIZE)
        print("In stpan:\n input_frames:\t", input_frames)

        # Estimated spatio-temporal flow
        if noise_type == "double":
            print("NOISE_TYPE: double")
            input_stack = tf.concat([tf.reshape(input_frames, [batch_size, H, W, C * T]), noise_level], axis=-1)
            op_flow, weights = flow_estimation_network_v2(tf.reshape(input_stack, [batch_size, H, W, C * T + 2]), ch0=64, N=4, D=3, norm_type=norm_type, anneal_term=anneal_term)
        elif noise_type == "single_std":
            print("NOISE_TYPE: single_std")
            input_stack = tf.concat([tf.reshape(input_frames, [batch_size, H, W, C * T]), noise_level], axis=-1)
            op_flow, weights = flow_estimation_network_v2(tf.reshape(input_stack, [batch_size, H, W, C * T + 1]), ch0=64, N=4, D=3, norm_type=norm_type, anneal_term=anneal_term)
        else:
            print("Blind!")
            input_stack = tf.reshape(input_frames, [batch_size, H, W, C * T])
            op_flow, weights = flow_estimation_network_v2(tf.reshape(input_frames, [batch_size, H, W, C * T]), ch0=64, N=4, D=3, norm_type=norm_type, anneal_term=anneal_term)

        # Feed image sequence and optical flow into spatio-temporal Sampler
        print("op_flow:\t", op_flow)
        CH_FLOW = KERNEL_K * KERNEL_K * KERNEL_T

        # Use the following code to replace the followed loop to accelerate
        # op_flow = tf.transpose(op_flow, [0, 3, 1, 2, 4])
        # op_flow = tf.reshape(op_flow, [B*CH_FLOW, H, W, 3])
        # ipf = tf.expand_dims(input_frames, axis=1)
        # ipf = tf.tile(ipf, [1, CH_FLOW, 1, 1, 1, 1])
        # ipf = tf.reshape(ipf, [B*CH_FLOW, H, W, T, C])
        # out = spatial_temporal_sampler(ipf, op_flow, T//2)
        # out = tf.reshape(out, [B, CH_FLOW, H, W, C])
        # sample_points = tf.transpose(out, [0, 2, 3, 1, 4])

        sample_points_stack = []
        for i in range(CH_FLOW):
            sample_points_stack.append(spatial_temporal_sampler(input_frames, op_flow[...,i,:], origin))
        sample_points = tf.stack(sample_points_stack, axis=-2)

        with tf.name_scope("w_predict_subnet"):
            points = tf.reshape(sample_points, [batch_size, H, W, C*CH_FLOW])
            weights = conv2d(tf.concat([weights, points, input_stack], axis=-1), 64, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
            weights = conv2d(weights, 64, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
            weights = conv2d(weights, 64, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
            weights = conv2d(weights, CH_FLOW, 3, padding="same", activation=None, norm_type=norm_type)

        weights = tf.expand_dims(weights, axis=-1)
        per_points_out = weights * sample_points
        out = tf.reduce_sum(per_points_out, axis=-2)

        return out, op_flow, weights, per_points_out

def unet_subset(inputs, ch, N, D=3, norm_type="none"):
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
    _inputs = inputs
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


def flow_estimation_network_v2(frames, anneal_term=1, ch0=64, N=4, D=3, norm_type="none"):
    """
    main net work for offset & weight estimation.
    input:
        frames: image frames [B, H, W, 3*T]
        anneal_term: DEPRECATED. use default 1.
        ch0: channel num of first layer.
        N: times for Encoder downsample
        D: num of conv laters per conv block
        norm_type: use default none
    output:
        flow: predicted offsets 
        weights: predicted weights
    """
    KERNEL_K = FLAGS.kernel_size_k
    KERNEL_T = FLAGS.kernel_size_t
    DILATE_RATE = FLAGS.kernel_dilate_rate
    MAX_OFFSET = FLAGS.max_offset
    FRAME_SIZE = FLAGS.frame_size
    
    with tf.name_scope("unet"):
        ch_flow = KERNEL_K * KERNEL_K * KERNEL_T
        inputs = frames
        for i in range(D):
            print('Pre-Layer with {} channels at N={}'.format(ch0, N))
            inputs = conv2d(inputs, ch0, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)

        inputs = tf.layers.average_pooling2d(inputs, pool_size=2, strides=2)
        ch = ch0 * 2
        # [B, H/2, W/2, 128]
        for i in range(D):
            print('Pre-Layer with {} channels at N={}'.format(ch, N))
            inputs = conv2d(inputs, ch, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
        # middle layers for unet: 256-512-512-512-256
        inputs = unet_subset(inputs, ch * 2, N - 2, D, norm_type=norm_type)
        inputs = conv2d(inputs, 256, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)

        flow_weights = inputs[..., : 128]
        flow = inputs[..., 128:]

        for i in range(D-1):
            flow_weights = conv2d(flow_weights, 128, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
            flow = conv2d(flow, 128, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)

        # resize output to original image and return the reshaped tensor
        
        flow = upsample_layer(flow, method=tf.image.ResizeMethod.BILINEAR)  # [B, H, W, 3*KERNEL_KERNEL_K^2]
        flow = conv2d(tf.concat([flow, frames], axis=-1), 128, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
        flow = conv2d(flow, 128, 3, padding="same", activation=tf.nn.relu, norm_type=norm_type)
        flow = conv2d(flow, ch_flow * 3, 3, padding="same", activation=None)

        flow_weights = upsample_layer(flow_weights, method=tf.image.ResizeMethod.BILINEAR)

        print("flow_weights shape : ", flow_weights.get_shape().as_list())
        print("flow shape", flow.get_shape().as_list())

        mid = KERNEL_K // 2
        flow_rebased = []

        flow = tf.nn.tanh(flow)
        flow = tf.reshape(flow, [tf.shape(frames)[0], tf.shape(frames)[1], tf.shape(frames)[2], ch_flow, 3])
        print("Using 3d %d points for flow" % ch_flow)
        mid_t = KERNEL_T//2
        for t in range(KERNEL_T):
            for i in range(KERNEL_K):
                for j in range(KERNEL_K):
                    u = flow[:, :, :, t * KERNEL_K * KERNEL_K + i * KERNEL_K + j, 0] * MAX_OFFSET + (j - mid) * DILATE_RATE
                    v = flow[:, :, :, t * KERNEL_K * KERNEL_K + i * KERNEL_K + j, 1] * MAX_OFFSET + (i - mid) * DILATE_RATE
                    w = flow[:, :, :, t * KERNEL_K * KERNEL_K + i * KERNEL_K + j, 2] * (FRAME_SIZE // 2) + t - mid_t
                    flow_rebased.append(tf.stack([u, v, w], axis=-1))
        flow = tf.stack(flow_rebased, axis=-2)
        return flow, flow_weights