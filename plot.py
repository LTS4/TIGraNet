#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
"""
    Plot functions to create figures.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import torch.nn as nn

from paths import FIGURES_DIR, DEBUG_DIR_MNIST_012

def plot_loss_and_acc(run_time, run_name, history):
    """Generate a plot with training loss and validation accuracy for a specific model."""

    num_epochs = list(range(len(history)))
    loss = [t[0] for t in history]
    acc = [t[1] for t in history]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    r1 = ax1.plot(num_epochs, loss, color='red', label='training loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    r2 = ax2.plot(num_epochs, acc, color='blue', label='validation accuracy')
    ax2.set_ylabel('accuracy')
    
    lns = r2 + r1
    labs = [l.get_label() for l in lns]
    leg = plt.legend(lns, labs, loc='center right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(FIGURES_DIR + run_time + '_' + run_name + '_loss_and_acc_results.png')
    plt.gcf().clear()

def plot_loss(run_time, run_name, history):
    """Generate a plot with training and validation loss for the given history."""

    num_epochs = list(range(len(history)))
    train_loss = [t[0] for t in history]
    valid_loss = [t[1] for t in history]

    fig, ax1 = plt.subplots()

    train_curve = ax1.plot(num_epochs, train_loss, color='red', label='training')
    valid_curve = ax1.plot(num_epochs, valid_loss, color='blue', label='validation')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    leg = plt.legend(loc='upper right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(FIGURES_DIR + run_time + '_' + run_name + '_losses.png')
    plt.gcf().clear()

def plot_error(run_time, run_name, history):
    """Generates a plot with training validation errors for the given history."""

    num_epochs = list(range(len(history)))
    train_errors = [t[0] for t in history]
    valid_errors = [t[1] for t in history]

    fig, ax1 = plt.subplots()

    train_curve = ax1.plot(num_epochs, train_errors, color='red', label='training')
    valid_curve = ax1.plot(num_epochs, valid_errors, color='blue', label='validation')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('error %')

    leg = plt.legend(loc='upper right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(FIGURES_DIR + run_time + '_' + run_name + '_errors.png')
    plt.gcf().clear()

def colorbar(mappable):
    """Create a colorbar that matches properly the size of the image."""
    
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    return fig.colorbar(mappable, cax=cax)

def show_tensor(image, filter_name, dim, num_nodes):
    """Show the image given by the tensor."""

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image.view(dim,dim), cmap='jet')

    # add suptitle and display the plot
    plt.suptitle(filter_name)
    plt.tight_layout(h_pad=1)
    plt.show()

def show_cheb_poly_tensor(cheb_poly, filter_name, num_filters, dim, num_nodes):
    """Show chebyshev polynomial of the intermediary layers given by the tensor."""

    # create the figure containing all the filters
    fig = plt.figure(figsize=(15,10))
    for i in range(num_filters):
        for j in range(10):
            ax = fig.add_subplot(num_filters, 10, i*10 + j+1) # this line adds sub-axes
            im = ax.imshow(cheb_poly.detach()[1, i, :, j].contiguous().view(dim,dim), cmap='jet')

    # add suptitle and display the plot
    plt.suptitle(filter_name)
    plt.tight_layout(h_pad=1)
    plt.show()