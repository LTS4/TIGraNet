#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This modules contains some testing of intermediary results between the PyTorch and the Theano framework.
"""

import numpy as np

from paths import DEBUG_DIR_MNIST_012, DEBUG_DIR_MNIST_rot, DEBUG_DIR_ETH80
from debug import *

######################################################################################################
#                                           PROCEDURE                                                #
######################################################################################################

# In order to correctly debug the PyTorch framework we load the exact same data, i.e. the first batch 
# of the test data. We then save/print and compare every step of the forward pass of a rather simple 
# model at the beginning, and we increase the complexity of the model as the tests are passed 
# correctly. The last 't' in files named like xxx_t.xxx stands for Theano, 'p' stands for PyTorch.


######################################################################################################
#                             WITH ALL WEIGHTS SET TO 1 - MNIST_012                                  #
######################################################################################################

path = DEBUG_DIR_MNIST_012 + 'constant_weights/'

# input averaged with batch ##########################################################################
prepared_input_p = np.load(path + 'prepared_input_p.npy')
prepared_input_t = np.load(path + 'prepared_input_t.npy')

prepared_input_t = np.transpose(prepared_input_t, (0,2,1))

# plot_pytorch_theano_image(
#     [prepared_input_p[0,:,0], prepared_input_t[0,:,0]],
#     dir=path,
#     name='prepared_input'
#     )
# np.testing.assert_allclose(actual=prepared_input_p[0], desired=prepared_input_t[0], rtol=1e-7) # OK
######################################################################################################

##################################
# Spectral Convolutional Layer 1 #
##################################

# filter operator ####################################################################################
filter_operator1_p = np.load(path + 'filter_operator1_p.npy')
filter_operator1_t = np.load(path + 'filter_operator1_t.npy')

filter_operator1_t = np.transpose(filter_operator1_t, (2,1,0))

# plot_pytorch_theano_filter_operator(
#     [filter_operator1_p[:,:,2], filter_operator1_t[:,:,2]],
#     dir=path,
#     name='filter_operator1'
#     )
# np.testing.assert_allclose(actual=filter_operator1_p, desired=filter_operator1_t, rtol=1e-7) # 0.17%
######################################################################################################

# y ##################################################################################################
y1_p = np.load(path + 'y1_p.npy')
y1_t = np.load(path + 'y1_t.npy')

y1_t = np.transpose(y1_t, (0,3,2,1))

# plot_pytorch_theano_image(
#     [y1_p[0,:,0], y1_t[0,:,0]],
#     dir=path,
#     name='y1'
#     )
# np.testing.assert_allclose(actual=y1_p, desired=y1_t, rtol=1e-6) # 3.30%
######################################################################################################

# z ##################################################################################################
spectral_conv1_p = np.load(path + 'spectral_conv1_p.npy')
spectral_conv1_t = np.load(path + 'spectral_conv1_t.npy')

spectral_conv1_t = np.transpose(spectral_conv1_t, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv1_p[0,:,0], spectral_conv1_t[0,:,0]],
#     dir=path,
#     name='spectral_conv1'
#     )

# plot_pytorch_theano_image_diff(
#     [spectral_conv1_p[0,:,0], spectral_conv1_t[0,:,0]],
#     dir=path,
#     name='spectral_conv1_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv1_p, desired=spectral_conv1_t, rtol=1e-6) # 3.30%
######################################################################################################

##################################
# Dynamic Pooling 1              #
##################################

# mask ###############################################################################################
mask1_p = np.load(path + 'mask1_p.npy')
mask1_t = np.load(path + 'mask1_t.npy')

mask1_t = mask1_t[..., np.newaxis]

# plot_pytorch_theano_image(
#     [mask1_p[0,:,0],
#     mask1_t[0,:,0]],
#     dir=path,
#     name='mask1'
#     )

# plot_pytorch_theano_image_diff(
#     [mask1_p[0,:,0], mask1_t[0,:,0]],
#     dir=path,
#     name='mask1_diff'
# ) 
# np.testing.assert_allclose(actual=mask1_p, desired=mask1_t, rtol=1e-6) # 1.41%
######################################################################################################

##################################
# Spectral Convolutional Layer 2 #
##################################

# y ##################################################################################################
y2_p = np.load(path + 'y2_p.npy')
y2_t = np.load(path + 'y2_t.npy')

# y2_t = np.transpose(y2_t, (0,3,2,1))
# plot_pytorch_theano_image(
#     [y2_p[0,:,0,0],
#     y2_t[0,:,0,0]],
#     dir=path,
#     name='y2'
#     )
# np.testing.assert_allclose(actual=y2_p, desired=y2_t, rtol=1e-6) # 9.68%
######################################################################################################

# z ##################################################################################################
spectral_conv2_p = np.load(path + 'spectral_conv2_p.npy')
spectral_conv2_t = np.load(path + 'spectral_conv2_t.npy')

spectral_conv2_t = np.transpose(spectral_conv2_t, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv2_p[0,:,0],
#     spectral_conv2_t[0,:,0]],
#     dir=path,
#     name='spectral_conv2'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv2_p[0,:,0], spectral_conv2_t[0,:,0]],
#     dir=path,
#     name='spectral_conv2_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv2_p, desired=spectral_conv2_t, rtol=1e-6) # 9.78%
######################################################################################################

##################################
# Dynamic Pooling 2              #
##################################

# mask ###############################################################################################
mask2_p = np.load(path + 'mask2_p.npy')
mask2_t = np.load(path + 'mask2_t.npy')

mask2_t = mask2_t[..., np.newaxis]

# plot_pytorch_theano_image(
#     [mask2_p[0,:,0],
#     mask2_t[0,:,0]],
#     dir=path,
#     name='mask2'
#     )
# plot_pytorch_theano_image_diff(
#     [mask2_p[0,:,0], mask2_t[0,:,0]],
#     dir=path,
#     name='mask2_diff'
# ) 
# np.testing.assert_allclose(actual=mask2_p, desired=mask2_t, rtol=1e-6) # 1.55%
######################################################################################################

##################################
# Spectral Convolutional Layer 3 #
##################################

# y ##################################################################################################
y3_p = np.load(path + 'y3_p.npy')
y3_t = np.load(path + 'y3_t.npy')

# y3_t = np.transpose(y3_t, (0,3,2,1))
# plot_pytorch_theano_image(
#     [y3_p[0,:,0,0],
#     y3_t[0,:,0,0]],
#     dir=path,
#     name='y3'
#     )
# np.testing.assert_allclose(actual=y3_p, desired=y3_t, rtol=1e-6) # 5.12%
######################################################################################################

# z ##################################################################################################
spectral_conv3_p = np.load(path + 'spectral_conv3_p.npy')
spectral_conv3_t = np.load(path + 'spectral_conv3_t.npy')

spectral_conv3_t = np.transpose(spectral_conv3_t, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv3_p[0,:,0],
#     spectral_conv3_t[0,:,0]],
#     dir=path,
#     name='spectral_conv3'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv3_p[0,:,0], spectral_conv3_t[0,:,0]],
#     dir=path,
#     name='spectral_conv3_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv3_p, desired=spectral_conv3_t, rtol=1e-6) # 5.09%
######################################################################################################

##################################
# Dynamic Pooling 3              #
##################################

# mask ###############################################################################################
mask3_p = np.load(path + 'mask3_p.npy')
mask3_t = np.load(path + 'mask3_t.npy')

mask3_t = mask3_t[..., np.newaxis]

# plot_pytorch_theano_image(
#     [mask3_p[0,:,0],
#     mask3_t[0,:,0]],
#     dir=path,
#     name='mask3'
#     )
# plot_pytorch_theano_image_diff(
#     [mask3_p[0,:,0], mask3_t[0,:,0]],
#     dir=path,
#     name='mask3_diff'
# ) 
# np.testing.assert_allclose(actual=mask3_p, desired=mask3_t, rtol=1e-6) # 0.34%
######################################################################################################

##################################
# Statitic                       #
##################################

# feature_vec ########################################################################################
statistic_p = np.load(path + 'statistic_p.npy')
statistic_t = np.load(path + 'statistic_t.npy')

# plot_pytorch_theano_statistic(
#     [statistic_p[0,:], statistic_t[0,:]],
#     shape=(20,9),
#     dir=path,
#     name='statistic'
#     )
# plot_pytorch_theano_statistic_diff(
#     [statistic_p[0,:], statistic_t[0,:]],
#     shape=(20,9),
#     dir=path,
#     name='statistic_diff'
# ) 
# np.testing.assert_allclose(actual=statistic_p, desired=statistic_t, rtol=1e-6) # 74.15%
######################################################################################################


######################################################################################################
#                        WITH PRETRAINED WEIGHTS LOADED - MNIST_012                                  #
######################################################################################################

path = DEBUG_DIR_MNIST_012 + 'pretrained_weights/'

##################################
# Spectral Convolutional Layer 1 #
##################################

# filter operator ####################################################################################
filter_operator1_p_pw = np.load(path + 'filter_operator1_p_pw.npy')
filter_operator1_t_pw = np.load(path + 'filter_operator1_t_pw.npy')

filter_operator1_t_pw = np.transpose(filter_operator1_t_pw, (2,1,0))

# plot_pytorch_theano_filter_operator(
#     [filter_operator1_p_pw[:,:,0],
#     filter_operator1_t_pw[:,:,0]],
#     dir=path,
#     name='filter_operator1'
#     )
# np.testing.assert_allclose(actual=filter_operator1_p_pw, desired=filter_operator1_t_pw, rtol=1e-6) # 0.07%
######################################################################################################

# y ##################################################################################################
y1_p_pw = np.load(path + 'y1_p_pw.npy')
y1_t_pw = np.load(path + 'y1_t_pw.npy')

y1_t_pw = np.transpose(y1_t_pw, (0,3,2,1))

# plot_pytorch_theano_image(
#     [y1_p_pw[0,:,0,0], y1_t_pw[0,:,0,0]],
#     dir=path,
#     name='y1'
#     )
# np.testing.assert_allclose(actual=y1_p_pw, desired=y1_t_pw, rtol=1e-6) # 1.49%
######################################################################################################

# z ##################################################################################################
spectral_conv1_p_pw = np.load(path + 'spectral_conv1_p_pw.npy')
spectral_conv1_t_pw = np.load(path + 'spectral_conv1_t_pw.npy')

spectral_conv1_t_pw = np.transpose(spectral_conv1_t_pw, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv1_p_pw[0,:,0], spectral_conv1_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv1'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv1_p_pw[0,:,0], spectral_conv1_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv1_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv1_p_pw, desired=spectral_conv1_t_pw, rtol=1e-6) # 1.49%
######################################################################################################

##################################
# Dynamic Pooling 1              #
##################################

# mask ###############################################################################################
mask1_p_pw = np.load(path + 'mask1_p_pw.npy')
mask1_t_pw = np.load(path + 'mask1_t_pw.npy')

mask1_t_pw = mask1_t_pw[...,np.newaxis]

# plot_pytorch_theano_image(
#     [mask1_p_pw[0,:,0], mask1_t_pw[0,:,0]],
#     dir=path,
#     name='mask1'
#     )
# plot_pytorch_theano_image_diff(
#     [mask1_p_pw[0,:,0], mask1_t_pw[0,:,0]],
#     dir=path,
#     name='mask1_diff'
# ) 
# np.testing.assert_allclose(actual=mask1_p_pw, desired=mask1_t_pw, rtol=1e-6) # 1.59%
######################################################################################################

##################################
# Spectral Convolutional Layer 2 #
##################################

# y ##################################################################################################
y2_p_pw = np.load(path + 'y2_p_pw.npy')
y2_t_pw = np.load(path + 'y2_t_pw.npy')

y2_t_pw = np.transpose(y2_t_pw, (0,3,2,1))

# plot_pytorch_theano_image(
#     [y2_p_pw[0,:,0,0], y2_t_pw[0,:,0,0]],
#     dir=path,
#     name='y2'
#     )
# np.testing.assert_allclose(actual=y2_p_pw, desired=y2_t_pw, rtol=1e-6) # 23.96%
######################################################################################################

# z ##################################################################################################
spectral_conv2_p_pw = np.load(path + 'spectral_conv2_p_pw.npy')
spectral_conv2_t_pw = np.load(path + 'spectral_conv2_t_pw.npy')

spectral_conv2_t_pw = np.transpose(spectral_conv2_t_pw, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv2_p_pw[0,:,0], spectral_conv2_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv2'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv2_p_pw[0,:,0], spectral_conv2_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv2_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv2_p_pw, desired=spectral_conv2_t_pw, rtol=1e-6) # 33.58%
######################################################################################################

##################################
# Dynamic Pooling 2              #
##################################

# mask ###############################################################################################
mask2_p_pw = np.load(path + 'mask2_p_pw.npy')
mask2_t_pw = np.load(path + 'mask2_t_pw.npy')

mask2_t_pw = mask2_t_pw[...,np.newaxis]

# plot_pytorch_theano_image(
#     [mask2_p_pw[0,:,0], mask2_t_pw[0,:,0]],
#     dir=path,
#     name='mask2'
#     )
# plot_pytorch_theano_image_diff(
#     [mask2_p_pw[0,:,0], mask2_t_pw[0,:,0]],
#     dir=path,
#     name='mask2_diff'
# ) 
# np.testing.assert_allclose(actual=mask2_p_pw, desired=mask2_t_pw, rtol=1e-6) # 0.80%
######################################################################################################

##################################
# Spectral Convolutional Layer 3 #
##################################

# z ##################################################################################################
spectral_conv3_p_pw = np.load(path + 'spectral_conv3_p_pw.npy')
spectral_conv3_t_pw = np.load(path + 'spectral_conv3_t_pw.npy')

spectral_conv3_t_pw = np.transpose(spectral_conv3_t_pw, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv3_p_pw[0,:,0], spectral_conv3_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv3'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv3_p_pw[0,:,0], spectral_conv3_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv3_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv3_p_pw, desired=spectral_conv3_t_pw, rtol=1e-6) # 37.04%
######################################################################################################

##################################
# Dynamic Pooling 3              #
##################################

# mask ###############################################################################################
mask3_p_pw = np.load(path + 'mask3_p_pw.npy')
mask3_t_pw = np.load(path + 'mask3_t_pw.npy')

mask3_t_pw = mask3_t_pw[...,np.newaxis]

# plot_pytorch_theano_image(
#     [mask3_p_pw[0,:,0], mask3_t_pw[0,:,0]],
#     dir=path,
#     name='mask3'
#     )
# plot_pytorch_theano_image_diff(
#     [mask3_p_pw[0,:,0], mask3_t_pw[0,:,0]],
#     dir=path,
#     name='mask3_diff'
# ) 
# np.testing.assert_allclose(actual=mask3_p_pw, desired=mask3_t_pw, rtol=1e-6) # 3.38%
######################################################################################################

##################################
# Statitic                       #
##################################

# feature_vec ########################################################################################
statistic_p_pw = np.load(path + 'statistic_p_pw.npy')
statistic_t_pw = np.load(path + 'statistic_t_pw.npy')

# plot_pytorch_theano_statistic(
#     [statistic_p_pw[0,:], statistic_t_pw[0,:]],
#     shape=(20,9),
#     dir=path,
#     name='statistic'
#     )
# plot_pytorch_theano_statistic_diff(
#     [statistic_p_pw[0,:], statistic_t_pw[0,:]],
#     shape=(20,9),
#     dir=path,
#     name='statistic_diff'
# ) 
# np.testing.assert_allclose(actual=statistic_p_pw, desired=statistic_t_pw, rtol=1e-6) # 100%
######################################################################################################


######################################################################################################
#                            WITH PRETRAINED WEIGHTS LOADED - MNIST_ROT                              #
######################################################################################################

path = DEBUG_DIR_MNIST_rot

##################################
# Spectral Convolutional Layer 1 #
##################################

# filter operator ####################################################################################
filter_operator1_p_pw = np.load(path + 'filter_operator1_p_pw.npy')
filter_operator1_t_pw = np.load(path + 'filter_operator1_t_pw.npy')

filter_operator1_t_pw = np.transpose(filter_operator1_t_pw, (2,1,0))

# plot_pytorch_theano_filter_operator(
#     [filter_operator1_p_pw[:,:,0],
#     filter_operator1_t_pw[:,:,0]],
#     dir=path,
#     name='filter_operator1'
#     )
# np.testing.assert_allclose(actual=filter_operator1_p_pw, desired=filter_operator1_t_pw, rtol=1e-6) # 0.66%
######################################################################################################

# z ##################################################################################################
spectral_conv1_p_pw = np.load(path + 'spectral_conv1_p_pw.npy')
spectral_conv1_t_pw = np.load(path + 'spectral_conv1_t_pw.npy')

spectral_conv1_t_pw = np.transpose(spectral_conv1_t_pw, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv1_p_pw[0,:,0], spectral_conv1_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv1'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv1_p_pw[0,:,0], spectral_conv1_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv1_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv1_p_pw, desired=spectral_conv1_t_pw, rtol=1e-6) # 56.14%
######################################################################################################

##################################
# Dynamic Pooling 1              #
##################################

# mask ###############################################################################################
mask1_p_pw = np.load(path + 'mask1_p_pw.npy')
mask1_t_pw = np.load(path + 'mask1_t_pw.npy')

mask1_t_pw = mask1_t_pw[...,np.newaxis]

# plot_pytorch_theano_image(
#     [mask1_p_pw[0,:,0], mask1_t_pw[0,:,0]],
#     dir=path,
#     name='mask1'
#     )
# plot_pytorch_theano_image_diff(
#     [mask1_p_pw[0,:,0], mask1_t_pw[0,:,0]],
#     dir=path,
#     name='mask1_diff'
# ) 
# np.testing.assert_allclose(actual=mask1_p_pw, desired=mask1_t_pw, rtol=1e-6) # 0%
######################################################################################################

##################################
# Spectral Convolutional Layer 2 #
##################################

# z ##################################################################################################
spectral_conv2_p_pw = np.load(path + 'spectral_conv2_p_pw.npy')
spectral_conv2_t_pw = np.load(path + 'spectral_conv2_t_pw.npy')

spectral_conv2_t_pw = np.transpose(spectral_conv2_t_pw, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv2_p_pw[0,:,0], spectral_conv2_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv2'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv2_p_pw[0,:,0], spectral_conv2_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv2_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv2_p_pw, desired=spectral_conv2_t_pw, rtol=1e-7) # 96.54%
######################################################################################################

##################################
# Dynamic Pooling 2              #
##################################

# mask ###############################################################################################
mask2_p_pw = np.load(path + 'mask2_p_pw.npy')
mask2_t_pw = np.load(path + 'mask2_t_pw.npy')

mask2_t_pw = mask2_t_pw[...,np.newaxis]

# plot_pytorch_theano_image(
#     [mask2_p_pw[0,:,0], mask2_t_pw[0,:,0]],
#     dir=path,
#     name='mask2'
#     )
# plot_pytorch_theano_image_diff(
#     [mask2_p_pw[0,:,0], mask2_t_pw[0,:,0]],
#     dir=path,
#     name='mask2_diff'
# ) 
# np.testing.assert_allclose(actual=mask2_p_pw, desired=mask2_t_pw, rtol=1e-6) # 0.99%
######################################################################################################

##################################
# Statitic                       #
##################################

# feature_vec ########################################################################################
statistic_p_pw = np.load(path + 'statistic_p_pw.npy')
statistic_t_pw = np.load(path + 'statistic_t_pw.npy')

# plot_pytorch_theano_statistic(
#     [statistic_p_pw[0,:], statistic_t_pw[0,:]],
#     shape=(20,28),
#     dir=path,
#     name='statistic'
#     )
# plot_pytorch_theano_statistic_diff(
#     [statistic_p_pw[0,:], statistic_t_pw[0,:]],
#     shape=(20,28),
#     dir=path,
#     name='statistic_diff'
# ) 
# np.testing.assert_allclose(actual=statistic_p_pw, desired=statistic_t_pw, rtol=1e-6) # 100%
######################################################################################################


######################################################################################################
#                            WITH PRETRAINED WEIGHTS LOADED - ETH80                                  #
######################################################################################################

path = DEBUG_DIR_ETH80

##################################
# Spectral Convolutional Layer 1 #
##################################

# filter operator ####################################################################################
filter_operator1_p_pw = np.load(path + 'filter_operator1_p_pw.npy')
filter_operator1_t_pw = np.load(path + 'filter_operator1_t_pw.npy')

filter_operator1_t_pw = np.transpose(filter_operator1_t_pw, (2,1,0))

# plot_pytorch_theano_filter_operator(
#     [filter_operator1_p_pw[:,:,0],
#     filter_operator1_t_pw[:,:,0]],
#     dir=path,
#     name='filter_operator1'
#     )
# np.testing.assert_allclose(actual=filter_operator1_p_pw, desired=filter_operator1_t_pw, rtol=1e-6) # 0.66%
######################################################################################################

# z ##################################################################################################
spectral_conv1_p_pw = np.load(path + 'spectral_conv1_p_pw.npy')
spectral_conv1_t_pw = np.load(path + 'spectral_conv1_t_pw.npy')

spectral_conv1_t_pw = np.transpose(spectral_conv1_t_pw, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv1_p_pw[0,:,0], spectral_conv1_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv1'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv1_p_pw[0,:,0], spectral_conv1_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv1_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv1_p_pw[:50,:,:], desired=spectral_conv1_t_pw, rtol=1e-6) # 49.58%
######################################################################################################

##################################
# Dynamic Pooling 1              #
##################################

# mask ###############################################################################################
mask1_p_pw = np.load(path + 'mask1_p_pw.npy')
mask1_t_pw = np.load(path + 'mask1_t_pw.npy')

mask1_t_pw = mask1_t_pw[...,np.newaxis]

# plot_pytorch_theano_image(
#     [mask1_p_pw[0,:,0], mask1_t_pw[0,:,0]],
#     dir=path,
#     name='mask1'
#     )
# plot_pytorch_theano_image_diff(
#     [mask1_p_pw[0,:,0], mask1_t_pw[0,:,0]],
#     dir=path,
#     name='mask1_diff'
# ) 
# np.testing.assert_allclose(actual=mask1_p_pw[:50,:,:], desired=mask1_t_pw, rtol=1e-6) # 0.80%
######################################################################################################

##################################
# Spectral Convolutional Layer 2 #
##################################

# z ##################################################################################################
spectral_conv2_p_pw = np.load(path + 'spectral_conv2_p_pw.npy')
spectral_conv2_t_pw = np.load(path + 'spectral_conv2_t_pw.npy')

spectral_conv2_t_pw = np.transpose(spectral_conv2_t_pw, (0,2,1))

# plot_pytorch_theano_image(
#     [spectral_conv2_p_pw[0,:,0], spectral_conv2_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv2'
#     )
# plot_pytorch_theano_image_diff(
#     [spectral_conv2_p_pw[0,:,0], spectral_conv2_t_pw[0,:,0]],
#     dir=path,
#     name='spectral_conv2_diff'
# ) 
# np.testing.assert_allclose(actual=spectral_conv2_p_pw[50:,:,:], desired=spectral_conv2_t_pw, rtol=1e-6) # 70.56%
######################################################################################################

##################################
# Dynamic Pooling 2              #
##################################

# mask ###############################################################################################
mask2_p_pw = np.load(path + 'mask2_p_pw.npy')
mask2_t_pw = np.load(path + 'mask2_t_pw.npy')

mask2_t_pw = mask2_t_pw[...,np.newaxis]

# plot_pytorch_theano_image(
#     [mask2_p_pw[0,:,0], mask2_t_pw[0,:,0]],
#     dir=path,
#     name='mask2'
#     )
# plot_pytorch_theano_image_diff(
#     [mask2_p_pw[0,:,0], mask2_t_pw[0,:,0]],
#     dir=path,
#     name='mask2_diff'
# )
# np.testing.assert_allclose(actual=mask2_p_pw[:50,:,:], desired=mask2_t_pw, rtol=1e-6) # 12.93%
######################################################################################################

##################################
# Statitic                       #
##################################

# feature_vec ########################################################################################
statistic_p_pw = np.load(path + 'statistic_p_pw.npy')
statistic_t_pw = np.load(path + 'statistic_t_pw.npy')

# plot_pytorch_theano_statistic(
#     [statistic_p_pw[0,:], statistic_t_pw[0,:]],
#     shape=(20,24),
#     dir=path,
#     name='statistic'
#     )
# plot_pytorch_theano_statistic_diff(
#     [statistic_p_pw[0,:], statistic_t_pw[0,:]],
#     shape=(20,24),
#     dir=path,
#     name='statistic_diff'
# ) 
# np.testing.assert_allclose(actual=statistic_p_pw[:50,:], desired=statistic_t_pw, rtol=1e-6) # 100%
######################################################################################################
