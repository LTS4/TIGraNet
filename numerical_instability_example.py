#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Illustrates numerical instability for similar function in Theano and PyTorch.
"""

import torch
import numpy as np
import theano
import theano.tensor as T
import os

from paths import *

torch.manual_seed(7)

def func_pytorch(a, b):

    a = a.permute(0,2,1).contiguous().view(-1,400)
    b = b.view(400,-1)

    return torch.matmul(a, b).view(100,5,400,10).permute(0,2,3,1)

def func_theano(a, b):
    return T.tensordot(a, b, [[2], [2]])

def func_numpy(a,b):
    return np.tensordot(a,b, [[2],[2]])

a = torch.randn(100,400,5)
b = torch.randn(400,400,10)

a_p = a.permute(0,2,1)
b_p = b.permute(2,1,0)

out_pytorch = func_pytorch(a, b)
out_numpy = np.transpose(func_numpy(a_p,b_p), (0,3,2,1))

out_true = np.transpose(func_theano(a_p, b_p).eval(), (0,3,2,1))


# from debug import plot_pytorch_theano_image, plot_pytorch_theano_image_diff
# from path import *

# print(out.shape, out_true[0,:,0,0].shape)

# plot_pytorch_theano_image(
#     images=[out[8,:,2,3].numpy(), out_true[8,:,2,3]],
#     dir=DEBUG_DIR_MNIST_012 + 'constant_weights/',
#     name='temp'
# )

# plot_pytorch_theano_image_diff(
#     images=[out[8,:,2,3].numpy(), out_true[8,:,2,3]],
#     dir=DEBUG_DIR_MNIST_012 + 'constant_weights/',
#     name='temp_diff'
# )

np.testing.assert_allclose(actual=out_numpy, desired=out_true, rtol=1e-7) # OK
np.testing.assert_allclose(actual=out_pytorch, desired=out_true, rtol=1e-7) # 76% mismatch
