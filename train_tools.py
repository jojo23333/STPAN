import tensorflow as tf

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def sRGBforward(x):
    b = .0031308
    gamma = 1./2.2
    # a = .055
    # k0 = 12.92
    a = 1./(1./(b**gamma*(1.-gamma))-1.)
    k0 = (1+a)*gamma*b**(gamma-1.)
    gammafn = lambda x : (1+a)*tf.pow(tf.maximum(x,b),gamma)-a
    # gammafn = lambda x : (1.-k0*b)/(1.-b)*(x-1.)+1.
    srgb = tf.where(x < b, k0*x, gammafn(x))
    k1 = (1+a)*gamma
    srgb = tf.where(x > 1, k1*x-k1+1, srgb)
    return srgb

def per_pixel_convolve(img_stack, img_filter, K=3):
    """
    Convolve in a per-pixel manner
    input: img_stack: [B, H, W, 3]
           img_filter: [B, H, W, K*K]
           K: 3 refer to kernel size
    """
    img_shape = tf.shape(img_stack)
    H = img_shape[1]
    W = img_shape[2]
    kpad = K // 2

    img = tf.pad(img_stack, [[0,0],[kpad,kpad],[kpad,kpad],[0,0]])
    img_stack = []
    for i in range(K):
        for j in range(K):
            img_stack.append(img[:,i:i+H,j:j+W,:])
    img_stack = tf.stack(img_stack, axis=-2)
    img_filter = tf.expand_dims(img_filter, axis=-1)
    filtered_img = tf.reduce_sum(img_stack * img_filter, axis=-2)
    return filtered_img

def img_loss(x, gt):
    return tf.losses.mean_squared_error(x, gt) # + gradient_loss(x, gt)

def gradient(imgs):
  return tf.stack([.5*(imgs[:,1:,:-1,:]-imgs[:,:-1,:-1,:]), .5*(imgs[:,:-1,1:,:]-imgs[:,:-1,:-1,:])], axis=-1)

def gradient_loss(guess, truth):
  return tf.reduce_mean(tf.abs(gradient(guess)-gradient(truth)))