#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Utilitary functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import logging

import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from configuration import *

random.seed(SEED)

logger = logging.getLogger(__name__)

def select(dataset, size, digits_to_keep, stratified_sampling=False):
    """Select randomly specific elements given by digits_to_keep."""
    
    len_dataset = len(dataset)
    indices = list(range(len_dataset))
    random_select_indices = []
    random.shuffle(indices)

    if stratified_sampling:
        num_classes = len(digits_to_keep)
        classes = [[] for _ in range(num_classes)]

        for i in indices:
            if dataset[i][1] in digits_to_keep:
                classe = dataset[i][1]
                classes[classe].append(i)

        for i in range(np.min([len(classes[0]), len(classes[1]), len(classes[2])])):
            for j in range(num_classes):
                if len(random_select_indices) < size:
                    random_select_indices.append(classes[j][i])
                else:
                    break

    else:
        for i in indices:
            if len(random_select_indices) < size and dataset[i][1] in digits_to_keep:
                random_select_indices.append(i)
        
    return random_select_indices

def train_valid_split(dataset, train_size):
    """Split the dataset into training and validaiton set."""
    
    len_dataset = len(dataset)
    indices = list(range(len_dataset))

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    return SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)

def train_valid_test_split(dataset, train_size, valid_size):
    """Split the dataset into training, validation and testing set."""
   
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices, valid_indices, test_indices = indices[:train_size], indices[train_size:train_size+valid_size], indices[train_size+valid_size:]

    return SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices), SubsetRandomSampler(test_indices)

def imshow_data_loader(data_loader, eth80_class_names=[]):
    """Show image provided by the data loader."""
    
    # get a batch of data
    inputs, classes = next(iter(data_loader))
    out = torchvision.utils.make_grid(tensor=inputs)

    # get the corresponding values
    if eth80_class_names:
        title = [eth80_class_names[x] for x in classes]
        mean = ETH80_MEAN
        std = ETH80_STD
    else:
        title = [x for x in classes]
        mean = MNIST_MEAN
        std = MNIST_STD

    # build the original image
    out = out.numpy().transpose((1, 2, 0))
    out = std * out + mean
    out = np.clip(out, 0, 1)

    # display it
    plt.imshow(out)
    plt.title(title)
    plt.show()

def show_spectrum(tensor, num_filters):
    """Show the spectrum of the spectral layer. """
    
    return NotImplemented

def snapshot(saved_model_dir, run_time, run_name, is_best, epoch, err_epoch, model_state_dict, optim_state_dict):
    """Save the model state."""
    
    complete_name = '{}{}_{}_{}_{:.2f}'.format(saved_model_dir, run_time, run_name, epoch, err_epoch)
    
    states = {
        'model': model_state_dict,
        'optimizer': optim_state_dict
        }

	# Save the model
    with open(complete_name + '.pt', 'wb') as f:
        torch.save(states, f)

def load_pretrained_model(saved_model_dir, run_name, model):
    """Load the specified model."""
    
    states = glob.glob(saved_model_dir + run_name)[0]

    if torch.cuda.is_available():
        checkpoint = torch.load(states)
    else:
        checkpoint = torch.load(states, map_location=lambda storage, loc: storage)
        
    model.load_state_dict(checkpoint['model'])
    model.optimizer.load_state_dict(checkpoint['optimizer'])

    logging.info('Loaded {} model.'.format(run_name))

    return model

def init_mask(num_nodes, batch_size):
    """Initialize the nodes of interest by including all the nodes of the graph."""
    mask = Variable(torch.ones(batch_size, num_nodes, 1)).to(DEVICE)
    return mask

def count_class_freq(loader, num_classes):
    """Return the frequency for each class from the loader."""
        
    t = np.zeros(num_classes)
    for _, target in loader:
        for c in target:
            t[c] +=1
    return t

def get_dim(data):
    """Get the dimension of the input image."""
    dim = len(data[0])
    return dim