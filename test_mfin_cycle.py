"""
Created on Feb, 2019

author: Lin Zhang
Computer Vision Lab, ETH Zurich
lin.zhang@vision.ee.ethz.ch
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']

import numpy as np
import scipy.io
import tensorflow as tf
import model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def readImages(path):    
    matfile = scipy.io.loadmat(path)
    images = matfile['refs']
    images = np.swapaxes(images,0,2)
    images = np.expand_dims(images,axis=3)
    return images


databasepath = 'examples/imgs.mat'

normalize = True
tf.set_random_seed(1.0)
imsim = 'l2'
lr = 1e-4


"""
create graph
"""

image_size = [None,224,224,1]
with tf.device('/gpu:0'):    
    interpolator = model.forward(image_size, imsim)
    optimizer = tf.train.AdamOptimizer(lr).minimize(interpolator['total_loss'])    
    
saver = tf.train.Saver(max_to_keep=2)
sess = tf.Session(config=config)

init_from_saved_model = True

if init_from_saved_model:
    
    if imsim == 'ssim':
        saver.restore(sess, "model/mfin_cycle_ssim")
    elif imsim == 'l2':
        saver.restore(sess, "model/mfin_cycle_l2")            
else:
    sess.run(tf.global_variables_initializer())
    
"""
test
"""          

tsi = readImages(databasepath)

tsik = tsi[0::2,:,:,:]
tsiu = tsi[1::2,:,:,:]
tsiu = np.delete(tsiu, (-1), axis=0)
tsiu = np.delete(tsiu, (0), axis=0)

if normalize is True:
    amin = np.percentile(tsik,2)
    amax = np.percentile(tsik,98)
    tsik = (tsik - amin) / (amax - amin)

tsik1 = tsik[0:tsik.shape[0]-3,:,:,:]
tsik2 = tsik[1:tsik.shape[0]-2,:,:,:]
tsik4 = tsik[2:tsik.shape[0]-1,:,:,:]
tsik5 = tsik[3:tsik.shape[0]-0,:,:,:]

h, x3_hat = sess.run([interpolator['h'], interpolator['y2']], \
                     feed_dict={interpolator['x1']: tsik1, interpolator['x2']: tsik2, \
                                interpolator['x4']: tsik4, interpolator['x5']: tsik5})

x3_hat = x3_hat * (amax - amin) + amin

p1 = h[-3]
p2 = h[-2]
p3 = h[-1]

rmse = np.sqrt(np.mean(np.square(tsiu - x3_hat),axis=(1,2,3)))
mae = np.mean(np.abs(tsiu - x3_hat),axis=(1,2,3))

print("RMSE:", str(np.mean(rmse)), "MAE:", str(np.mean(mae)))

_dict = {}
_dict['x3_hat'] = x3_hat
_dict['p1'] = p1
_dict['p2'] = p2
result_name = 'examples/mfin_cycle_results.mat'
scipy.io.savemat(result_name,_dict)


        

