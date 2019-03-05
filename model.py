"""
Created on Feb, 2019

author: Lin Zhang
Computer Vision Lab, ETH Zurich
lin.zhang@vision.ee.ethz.ch
"""
"""
References
[1] https://github.com/daviddao/spatial-transformer-tensorflow
[2] https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
"""

import numpy as np
import tensorflow as tf


def _tf_fspecial_gauss(size, sigma):
    """
    copied from [2]
    """
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    """
    copied from [2]
    """
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  
def weight_variable(shape):
    std = np.sqrt(2/(shape[0]*shape[1]*shape[2]))
    initial = tf.random_normal(shape, stddev=std)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def upsample_layer(input_layer, w, b):
    height = input_layer.shape[1]
    width =  input_layer.shape[2]
    h_upsample = tf.image.resize_images(input_layer, [2*int(height), 2*int(width)])
    h_conv1 = tf.nn.relu(conv2d(h_upsample, w) + b)
    return h_conv1


def initial_weights():
    n_basic_convolution = 5    
    n_filters_convolution = [4,16,32,64,64,32]
    f_size = [7, 5, 3, 3, 3]
    
    weights = []
    biases = []
    
    for l in range(n_basic_convolution):
        w_size = [f_size[l],f_size[l],n_filters_convolution[l],n_filters_convolution[l+1]]

        w = weight_variable(w_size)
        b = bias_variable([n_filters_convolution[l+1]])
        weights.append(w)
        biases.append(b)
            
    n_subnet = 3
    n_filters_subnet = [32, 32, 16, 2]
    
    for l in range(n_subnet):
        w_size = [3,3, n_filters_subnet[l], n_filters_subnet[l+1]]
        
        w1 = weight_variable(w_size)
        b1 = bias_variable([n_filters_subnet[l+1]])
        w2 = weight_variable(w_size)
        b2 = bias_variable([n_filters_subnet[l+1]])
        w3 = weight_variable(w_size)
        b3 = bias_variable([n_filters_subnet[l+1]])
        
        weights.append(w1)
        weights.append(w2)
        weights.append(w3)

        biases.append(b1)
        biases.append(b2)
        biases.append(b3)
                    
    return weights, biases


def layer_evaluation(input_layer, weights, biases):
    h = []
    n_convolution = 3
    n_deconvolution = 2
    h_current = input_layer
    
    for l in range(n_convolution):
        w = weights[l]
        b = biases[l]
        h1 = tf.nn.relu(conv2d(h_current, w) + b)
        h.append(h1)
        h_current = avg_pool_2x2(h1)        
        h.append(h_current)
    
    for l in range(n_deconvolution):
        w = weights[l+3]
        b = biases[l+3]
        h_current = upsample_layer(h_current, w, b)
        h.append(h_current)

    
    h_output = h_current
    
    #subnet1
    h_current = tf.nn.relu(conv2d(h_output, weights[5]) + biases[5])
    h.append(h_current)
    h_current = tf.nn.relu(conv2d(h_current, weights[8]) + biases[8])
    h.append(h_current)
    h_current = tf.image.resize_images(h_current, [224, 224])
    h_p1 = tf.add(tf.nn.conv2d(h_current, weights[11], strides=[1, 1, 1, 1], padding='SAME'), biases[11])
    
    
    #subnet2
    h_current = tf.nn.relu(conv2d(h_output, weights[6]) + biases[6])
    h.append(h_current)
    h_current = tf.nn.relu(conv2d(h_current, weights[9]) + biases[9])
    h.append(h_current)
    h_current = tf.image.resize_images(h_current, [224, 224])
    h_p2 = tf.add(tf.nn.conv2d(h_current, weights[12], strides=[1, 1, 1, 1], padding='SAME'), biases[12])
    
    #subnet3
    h_current = tf.nn.relu(conv2d(h_output, weights[7]) + biases[7])
    h.append(h_current)
    h_current = tf.nn.relu(conv2d(h_current, weights[10]) + biases[10])
    h.append(h_current)
    h_current = tf.image.resize_images(h_current, [224, 224])
    h_p3 = tf.add(tf.nn.conv2d(h_current, weights[13], strides=[1, 1, 1, 1], padding='SAME'), biases[13])
    
    
    h.append(h_p1)
    h.append(h_p2)
    h.append(h_p3)
    
    return h


def grid_generator(p, mode):

    num_batch = tf.shape(p)[0]
    
    x = tf.linspace(0.0, 223.0, 224)
    y = tf.linspace(0.0, 223.0, 224)
    
    if mode == 'xy':
        x_t, y_t = tf.meshgrid(x, y)
    elif mode == 'ij':
        x_t, y_t = tf.meshgrid(x, y, indexing = 'ij')
    
    sampling_grid = tf.stack([x_t, y_t], axis=-1)
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1, 1]))
    
    sampling_grid = tf.add(sampling_grid, p)

    return sampling_grid


def get_pixel_value(img, x, y, mode):
    
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.cast(batch_idx, 'float32')
    b = tf.tile(batch_idx, (1, height, width))
    b = tf.cast(b, 'int32')

    if mode == 'xy':
        indices = tf.stack([b, y, x], 3)
    elif mode == 'ij':
        indices = tf.stack([b, x, y], 3)

    return tf.gather_nd(img, indices)


def bilinear_interpolator(img, p, mode):
    """
    adapted from [1]
    """
    
    max_y = tf.cast(224 - 1, 'int32')
    max_x = tf.cast(224 - 1, 'int32')
    zero = tf.zeros([], dtype=tf.int32)
    
    sampling_grid = grid_generator(p, mode)
    x = sampling_grid[:,:,:,0]
    y = sampling_grid[:,:,:,1]
    
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    

    Ia = get_pixel_value(img, x0, y0, mode)
    Ib = get_pixel_value(img, x0, y1, mode)
    Ic = get_pixel_value(img, x1, y0, mode)
    Id = get_pixel_value(img, x1, y1, mode)
         
    
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    
    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)
    
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    
    return out

def compute_grad(im):
    grad_dy = im[:,1:,:,:] - im[:,:-1,:,:]
    grad_dx = im[:,:,1:,:] - im[:,:,:-1,:]
    return grad_dx, grad_dy

def compute_smooth_loss(flowU, flowV):
    flow_gradU_dx, flow_gradU_dy = compute_grad(flowU)
    flow_gradV_dx, flow_gradV_dy = compute_grad(flowV)
    loss = tf.reduce_mean(tf.abs(flow_gradU_dx)) + tf.reduce_mean(tf.abs(flow_gradU_dy)) \
         +  tf.reduce_mean(tf.abs(flow_gradV_dx)) + tf.reduce_mean(tf.abs(flow_gradV_dy))
    return loss


def tranformation_composition(p1, p2):
    p_new1 = bilinear_interpolator(tf.expand_dims(p2[:,:,:,0], axis=3), p1, 'ij')
    p_new2 = bilinear_interpolator(tf.expand_dims(p2[:,:,:,1], axis=3), p1, 'ij')    
    return tf.concat([p_new1, p_new2], axis=-1) + p1


def compute_consistency_loss(p1, p2):
    cost_x = tf.reduce_mean(tf.square(p1[:,:,:,0] - p2[:,:,:,0]))
    cost_y = tf.reduce_mean(tf.square(p1[:,:,:,1] - p2[:,:,:,1]))
    return tf.add(cost_x, cost_y)


def forward(input_shape, imsim):

    x1 = tf.placeholder(tf.float32, input_shape, name='x1')
    x2 = tf.placeholder(tf.float32, input_shape, name='x2')
    x3 = tf.placeholder(tf.float32, input_shape, name='x3')
    x4 = tf.placeholder(tf.float32, input_shape, name='x4')
    x5 = tf.placeholder(tf.float32, input_shape, name='x5')
                 
    x = tf.concat([x1, x2, x4, x5], 3)
    
    w, b = initial_weights()
        
    h = layer_evaluation(x,w,b)
    
    flow3 = h[-1]
    flow2 = h[-2]
    flow1 = h[-3]
    
    
    x2_interp = bilinear_interpolator(x2, flow1, 'xy')
    x4_interp = bilinear_interpolator(x4, flow2, 'xy')
    
    x3_interp = bilinear_interpolator(x2, flow3, 'xy')
    
    flow_new = tranformation_composition(flow2, flow3)
            
    # cost - reconstruction error
    cost1 = 1 - tf_ssim(x3, x2_interp)
    cost2 = 1 - tf_ssim(x3, x4_interp)
    cost3 = 1 - tf_ssim(x4, x3_interp)
    
    flow1U = tf.expand_dims(flow1[:,:,:,0], axis=3)
    flow1V = tf.expand_dims(flow1[:,:,:,1], axis=3)
    flow2U = tf.expand_dims(flow2[:,:,:,0], axis=3)
    flow2V = tf.expand_dims(flow2[:,:,:,1], axis=3)
    flow3U = tf.expand_dims(flow3[:,:,:,0], axis=3)
    flow3V = tf.expand_dims(flow3[:,:,:,1], axis=3)
    
    smooth_loss = compute_smooth_loss(flow1U, flow1V) +\
    compute_smooth_loss(flow2U, flow2V) + compute_smooth_loss(flow3U, flow3V)
    consist_loss = compute_consistency_loss(flow1, flow_new)
    
    if imsim == 'ssim':
        cost1 = 1 - tf_ssim(x3, x2_interp)
        cost2 = 1 - tf_ssim(x3, x4_interp)
        cost3 = 1 - tf_ssim(x4, x3_interp)
        total_loss = cost1 + cost2 + cost3 + 0.1*smooth_loss + 0.05*consist_loss
        
    elif imsim == 'l2':
        cost1 = tf.reduce_mean(tf.square(x2_interp - x3))
        cost2 = tf.reduce_mean(tf.square(x4_interp - x3))
        cost3 = tf.reduce_mean(tf.square(x3_interp - x4))
        total_loss = cost1 + cost2 + cost3 + 0.001*smooth_loss + 0.0005*consist_loss
           
    return {'x1': x1, 'x2': x2, 'x3': x3,'x4': x4, 'x5': x5, 'y1': x2_interp, \
            'y2': x4_interp, 'y3': x3_interp,'h' : h, 'W' : w, 'cost1': cost1,\
            'cost2': cost2, 'cost3': cost3, 'smooth': smooth_loss, \
            'total_loss': total_loss, 'consist': consist_loss}

        