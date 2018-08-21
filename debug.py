#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Useful debugging functions.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from layers import SpectralConv
from paths import DEBUG_DIR_MNIST_012, DEBUG_DIR_MNIST_rot, DEBUG_DIR_ETH80, FIGURES_DIR
from plot import show_filters_single_window, colorbar

titles = ['PyTorch', 'Theano']
size_dict = {DEBUG_DIR_MNIST_012 + 'constant_weights/':20, DEBUG_DIR_MNIST_012 + 'pretrained_weights/':20, DEBUG_DIR_MNIST_rot:26, DEBUG_DIR_ETH80:50}

def get_min_max(images):
    """Return the min and max pixel values for a list of images.""" 
    vmin = []
    vmax = []
    for i in images:
        vmin.append(np.min(i))
        vmax.append(np.max(i))
    return np.min(vmin), np.max(vmax)

def plot_pytorch_theano_image(images, dir, name='temp'):
    """Plot and save image comparison between PyTorch and Theano framework."""
    size = size_dict[dir]
    fig = plt.figure(figsize=(12,5))

    vmin, vmax = get_min_max(images)

    delta = 1e-0
    if vmin==vmax:
        vmin -= delta

    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        im = ax.imshow(images[i].astype(int).reshape(size,size), cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
    
    colorbar(im)

    plt.tight_layout(h_pad=1)
    plt.savefig(dir + 'figures/' + name + '.pdf')
    #plt.show()

def plot_pytorch_theano_image_diff(images, dir, name='temp'):
    """Plot and save image comparison between PyTorch and Theano framework."""
    size = size_dict[dir]
    fig = plt.figure(figsize=(6,5))

    diff = np.abs(images[0]-images[1])

    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(diff.reshape(size,size), cmap='jet')
    ax.set_axis_off()
    colorbar(im)

    plt.tight_layout(h_pad=1)
    plt.savefig(dir + 'figures/' + name + '.pdf')

def plot_pytorch_theano_statistic(images, shape, dir, name='temp'):
    """Plot and save feature vector (from statistical layer) comparison between PyTorch and Theano framework."""
    h, w = shape
    if w==28:
        fig = plt.figure(figsize=(10,3))
    elif w==24:
        fig = plt.figure(figsize=(8,3))
    else:
        fig = plt.figure(figsize=(6,5))

    vmin, vmax = get_min_max(images)

    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        im = ax.imshow(images[i].reshape(h,w), cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
    
    colorbar(im)

    plt.tight_layout(h_pad=1)
    plt.savefig(dir + 'figures/' + name + '.pdf')
    #plt.show()

def plot_pytorch_theano_statistic_diff(images, shape, dir, name='temp'):
    """Plot and save feature vector (from statistical layer) comparison between PyTorch and Theano framework."""
    h, w = shape
    if w==28:
        #MNIST_ROT
        fig = plt.figure(figsize=(5,4))
    elif w==24:
        #ETH80
        fig = plt.figure(figsize=(5,4))
    else:
        fig = plt.figure(figsize=(3,5))

    diff = np.abs(images[0]-images[1])

    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(diff.reshape(h,w), cmap='jet')
    ax.set_axis_off()
    colorbar(im)
    
    plt.tight_layout(h_pad=1)
    plt.savefig(dir + 'figures/' + name + '.pdf')
    #plt.show()

def plot_pytorch_theano_filter_operator(images, dir, name='temp'):
    """Plot and save filter operator (from spectral conv layer) comparison between PyTorch and Theano framework."""
    size = size_dict[dir]
    fig = plt.figure(figsize=(15,10))
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        im = ax.imshow(images[i].reshape(size**2,size**2), cmap='jet')
        ax.set_title(titles[i])
        ax.set_axis_off()
        colorbar(im)

    plt.suptitle('filter operator')
    plt.tight_layout(h_pad=1)
    plt.savefig(dir + 'figures/' + name + '.pdf')
    plt.show()

def init_weights_constant(model, constant=1):
    """Initialize weights of the model with the constant passed in argument."""
    for m in model.modules():
        if isinstance(m, SpectralConv):
            nn.init.constant_(m.alpha.weight, constant)
            nn.init.constant_(m.beta.weight, constant)
        elif isinstance(m, nn.Sequential):
            for m in m.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant)
                    nn.init.constant_(m.bias, constant)

def display_weights(model):
    """Display the weights of the model."""
    for name, param in model.named_parameters():
        print('- - - - - - - - - - - - - - - - - - - \n{} \n\n{} \n\n{} \n- - - - - - - - - - - - - - - - - - - '.format(name, param, param.size()))

def plot_dataset():
    """Plot the training, validation and testing datasets for the mnist_rot, mnist_trans and eth80 datasets."""
    num_images = 6
    dataset_shapes = {'mnist_rot':26, 'mnist_trans':30, 'eth80':50}

    for dataset_name in ['mnist_rot', 'mnist_trans', 'eth80']:
        dataset_shape = dataset_shapes[dataset_name]
        
        fig = plt.figure(figsize=(10,7))
        fig.suptitle(t='train, valid, test images for ' + dataset_name)
        for i, d in enumerate(['train', 'val', 'test']):
            data = np.load('{}{}/dataset/{}_{}_signals.npy'.format(SAVED_DATA, dataset_name, dataset_name, d))
            labels = np.load('{}{}/dataset/{}_{}_labels.npy'.format(SAVED_DATA, dataset_name, dataset_name, d))

            for j in range(num_images):
                ax = fig.add_subplot(3, num_images, i*6 + j+1) # this line adds sub-axes
                im = ax.imshow(data[j].reshape(dataset_shape,dataset_shape), cmap='jet')
                ax.set_axis_off()
        plt.show()