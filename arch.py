import tensorflow as tf
from tensorflow.keras.layers import dot
import numpy as np

def norm(x, norm_type, is_train, G=32, eps=1e-5):
    with tf.variable_scope('%s_norm' % norm_type):
        if norm_type == "none":
            return x
        elif norm_type == 'batch':
            return tf.layers.batch_normalization(x, center=True, scale=True, training=is_train, epsilon=eps)
        elif norm_type == 'group':
            B, H, W, C  = x.get_shape().as_list()
            G = min(G, C)
            x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.reshape(x, [-1, G, C//G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            print("Group norm shape:", x.get_shape())

            x = (x - mean) / tf.sqrt(var + eps)
            gamma = tf.Variable(tf.constant(1.0, shape=[1, C, 1, 1]), dtype=tf.float32, name="gamma")
            beta = tf.Variable(tf.constant(1.0, shape=[1, C, 1, 1]), dtype=tf.float32, name="beta")
            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

            return tf.transpose(output, [0, 2, 3, 1])
        else: 
            raise NotImplementedError


def conv2d(input, output_ch, k=3, s=1, padding="same", activation=None, norm_type="none", is_train=True):
    output = tf.layers.conv2d(input, output_ch, k, padding=padding, activation=activation)
    if norm_type == "group":
        return tf.contrib.layers.group_norm(output)
    else:
        return output
    # return norm(output, norm_type, is_train)


def upsample_layer(ip, method=tf.image.ResizeMethod.BILINEAR):
    B, H, W, C = ip.get_shape().as_list()
    return tf.image.resize_images(ip, [H * 2, W * 2], method=method)


def residual_block_1(input, output_ch, is_train, k=3, s=1, name="residual", activation=tf.nn.relu, norm_type="batch"):
    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            f = conv2d(input, output_ch, k, s=s, activation=activation, norm_type=norm_type)
        with tf.variable_scope('res2'):
            f = conv2d(f, output_ch, k, s=s, activation=None, norm_type="none")
        f = activation(f + input)
        f = norm(f, norm_type, is_train)
    return f


def get_pixel_value(img, x, y, t):
    """
    Utility function to get pixel value for coordinate
    vectors x, y, t from a 5D tensor image sequence

    Input:
    img: input frames of img: [B, H, W, T, 3]
    x: x axis: [B, H, W]
    y: y axis: [B, H, W]
    t: T axis: [B, H, W]

    Output:
    corresponding values
    [B, H, W, 3]
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    # get b with shape [B, H, W] correspond to the batch id of each pixel in each image
    batch_idx = tf.range(batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size,1,1))
    b = tf.tile(batch_idx, (1, height, width))
    # stack batchid, y, x, t to get a list of coordinate
    indices = tf.stack([b, y, x, t], 3)

    return tf.gather_nd(img, indices)
    
    
def get_pixel_value_2d(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x, y, t from a 5D tensor image sequence

    Input:
    img: input frames of img: [B, H, W, 3]
    x: x axis: [B, H, W]
    y: y axis: [B, H, W]

    Output:
    corresponding values
    [B, H, W, 3]
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    # get b with shape [B, H, W] correspond to the batch id of each pixel in each image
    batch_idx = tf.range(batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size,1,1))
    b = tf.tile(batch_idx, (1, height, width))
    # stack batchid, y, x, t to get a list of coordinate
    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def spatial_sampler(input_frames, op_flow, t_origin):
    """
    This function perform as spatio-temporal sampler
    
    Input:
    input_frames: [B, H, W, C]
    op_flow: [B, H, W, C]
    t_origin: predicted frame id 

    Output:
    generated img yt_
    [B, H, W, C]
    """
    with tf.name_scope("sts"):
        H = tf.shape(input_frames)[1]
        W = tf.shape(input_frames)[2]

        max_y = tf.cast(H+1, 'int32')
        max_x = tf.cast(W+1, 'int32')
        zero = tf.zeros([H, W], dtype='int32')

        # Generate 
        x_range = tf.range(W)
        y_range = tf.range(H)
        x_t, y_t = tf.meshgrid(x_range, y_range)
        x_t = tf.expand_dims(x_t, 0)
        y_t = tf.expand_dims(y_t, 0)
        
        # opy_flow --> [B, H, W, 3]
        u = op_flow[:,:,:,0]
        v = op_flow[:,:,:,1]

        # x,y,t --> [B, H, W]
        x = tf.cast( x_t, 'float32') + u + 1.
        y = tf.cast( y_t, 'float32') + v + 1.

        # grab 8 nearest corner points for each (x_i, y_i, t_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # clip to valid range
        x0 = tf.clip_by_value(x0, zero, max_x) 
        x1 = tf.clip_by_value(x1, zero, max_x) 
        y0 = tf.clip_by_value(y0, zero, max_y) 
        y1 = tf.clip_by_value(y1, zero, max_y)

        input_frames = tf.pad(input_frames, [[0,0],[1,1],[1,1],[0,0]], "SYMMETRIC")

        I0 = tf.cast(get_pixel_value_2d(input_frames, x0, y0), 'float32')
        I1 = tf.cast(get_pixel_value_2d(input_frames, x1, y0), 'float32')
        I2 = tf.cast(get_pixel_value_2d(input_frames, x0, y1), 'float32')
        I3 = tf.cast(get_pixel_value_2d(input_frames, x1, y1), 'float32')

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        # W are all [B, H, W]
        w0 = (x1-x) * (y1-y) 
        w1 = (x-x0) * (y1-y) 
        w2 = (x1-x) * (y-y0) 
        w3 = (x-x0) * (y-y0) 

        # Expand to [B, H, W, 1]
        w0 = tf.expand_dims(w0, axis=3)
        w1 = tf.expand_dims(w1, axis=3)
        w2 = tf.expand_dims(w2, axis=3)
        w3 = tf.expand_dims(w3, axis=3)

        #compute output
        out = tf.add_n([w0*I0, w1*I1, w2*I2, w3*I3])
        return out



def spatial_temporal_sampler(input_frames, op_flow, t_origin):
    """
    This function perform as spatio-temporal sampler
    
    Input:
    input_frames: [B, H, W, T, C]
    op_flow: [B, H, W, C]
    t_origin: predicted frame id 

    Output:
    generated img yt_
    [B, H, W, C]
    """
    with tf.name_scope("sts"):
        H = tf.shape(input_frames)[1]
        W = tf.shape(input_frames)[2]
        T = tf.shape(input_frames)[3]

        max_y = tf.cast(H+1, 'int32')
        max_x = tf.cast(W+1, 'int32')
        max_t = tf.cast(T-1, 'int32')
        zero = tf.zeros([H, W], dtype='int32')

        # Generate 
        x_range = tf.range(W)
        y_range = tf.range(H)
        x_t, y_t = tf.meshgrid(x_range, y_range)
        t_t = tf.ones([H, W], dtype='int32') * t_origin
        x_t = tf.expand_dims(x_t, 0)
        y_t = tf.expand_dims(y_t, 0)
        t_t = tf.expand_dims(t_t, 0)
        
        # opy_flow --> [B, H, W, 3]
        u = op_flow[:,:,:,0]
        v = op_flow[:,:,:,1]
        w = op_flow[:,:,:,2]

        # x,y,t --> [B, H, W]
        x = tf.cast( x_t, 'float32') + u + 1.
        y = tf.cast( y_t, 'float32') + v + 1.
        t = tf.cast( t_t, 'float32') + w

        # grab 8 nearest corner points for each (x_i, y_i, t_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        t0 = tf.cast(tf.floor(t), 'int32')
        t1 = t0 + 1

        # clip to valid range
        x0 = tf.clip_by_value(x0, zero, max_x) 
        x1 = tf.clip_by_value(x1, zero, max_x) 
        y0 = tf.clip_by_value(y0, zero, max_y) 
        y1 = tf.clip_by_value(y1, zero, max_y) 
        t0 = tf.clip_by_value(t0, zero, max_t) 
        t1 = tf.clip_by_value(t1, zero, max_t)

        input_frames = tf.pad(input_frames, [[0,0],[1,1],[1,1],[0,0],[0,0]], "SYMMETRIC")

        I0 = tf.cast(get_pixel_value(input_frames, x0, y0, t0), 'float32')
        I1 = tf.cast(get_pixel_value(input_frames, x0, y0, t1), 'float32')
        I2 = tf.cast(get_pixel_value(input_frames, x0, y1, t0), 'float32')
        I3 = tf.cast(get_pixel_value(input_frames, x0, y1, t1), 'float32')
        I4 = tf.cast(get_pixel_value(input_frames, x1, y0, t0), 'float32')
        I5 = tf.cast(get_pixel_value(input_frames, x1, y0, t1), 'float32')
        I6 = tf.cast(get_pixel_value(input_frames, x1, y1, t0), 'float32')
        I7 = tf.cast(get_pixel_value(input_frames, x1, y1, t1), 'float32')

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')
        t0 = tf.cast(t0, 'float32')
        t1 = tf.cast(t1, 'float32')

        # W are all [B, H, W]
        w0 = (x1-x) * (y1-y) * (t1-t)
        w1 = (x1-x) * (y1-y) * (t-t0)
        w2 = (x1-x) * (y-y0) * (t1-t)
        w3 = (x1-x) * (y-y0) * (t-t0)
        w4 = (x-x0) * (y1-y) * (t1-t)
        w5 = (x-x0) * (y1-y) * (t-t0)
        w6 = (x-x0) * (y-y0) * (t1-t)
        w7 = (x-x0) * (y-y0) * (t-t0)

        # Expand to [B, H, W, 1]
        w0 = tf.expand_dims(w0, axis=3)
        w1 = tf.expand_dims(w1, axis=3)
        w2 = tf.expand_dims(w2, axis=3)
        w3 = tf.expand_dims(w3, axis=3)
        w4 = tf.expand_dims(w4, axis=3)
        w5 = tf.expand_dims(w5, axis=3)
        w6 = tf.expand_dims(w6, axis=3)
        w7 = tf.expand_dims(w7, axis=3)

        #compute output
        out = tf.add_n([w0*I0, w1*I1, w2*I2, w3*I3,
                        w4*I4, w5*I5, w6*I6, w7*I7])
        return out


def none_local_blocks_spatial(ip, compression=1, inter_dimension=256):
    """
    Module for spatial none local blocks using embeded Gaussian
    See the original paper: Non-local Neural Networks CVPR2018

    input: spatial input share be B, H, W, CH
    mode: 
    inter_dimension: 
    """
    B = tf.shape(ip)[0]
    H = tf.shape(ip)[1]
    W = tf.shape(ip)[2]
    # CH = tf.shape(ip)[3]
    CH = inter_dimension * 2

    x1 = tf.layers.conv2d(ip, inter_dimension, 1, padding="same", activation=None)
    x1 = tf.reshape(x1, [B, -1, inter_dimension])

    x2 = tf.layers.conv2d(ip, inter_dimension, 1, padding="same", activation=None)
    if compression > 1:
        x2 = tf.layers.max_pooling2d(x2, pool_size=compression, strides=compression)
    x2 = tf.reshape(x2, [B, -1, inter_dimension])


    f = dot([x1, x2], axes=2)
    print("F shape", f.get_shape())
    f = tf.nn.softmax(f)
    print("F after softmax shape", f.get_shape())

    g = tf.layers.conv2d(ip, inter_dimension, 1, padding="Same", activation=None)
    if compression > 1:
        g = tf.layers.max_pooling2d(g, pool_size=compression, strides=compression)
    g = tf.reshape(g, [B, -1, inter_dimension])

    y = dot([f, g], axes=[2,1])
    y = tf.reshape(y, [B, H, W, inter_dimension])
    y = tf.layers.conv2d(y, CH, 1, padding="same", activation=None)

    out = tf.concat([ip, y], axis=-1)
    out = tf.layers.conv2d(out, CH, 1, padding="same", activation=tf.nn.relu)
    return out