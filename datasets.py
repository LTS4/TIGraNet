#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Datasets module.
"""

import numpy as np
import logging

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from loader import MNIST_bis
from custom_transform import RandomTranslation, Substract
from utils import train_valid_split, train_valid_test_split
from configuration import *
from utils import get_dim, count_class_freq

logger = logging.getLogger(__name__)

torch.manual_seed(SEED)

# datasets mean and standard deviation used for normalization
# L = R * 299/1000 + G * 587/1000 + B * 114/1000
MNIST_MEAN = [0.458]
MNIST_STD = [0.225]
ETH80_MEAN = [0.426]
ETH80_STD = [0.166]

def load_dataset(dataset, train_size, valid_size, test_size):
    """Load the dataset passed in argument with the corresponding sizes for the training, validation and testing set."""

    if dataset == 'mnist_012':
        root = './data/mnist'
        num_classes = 3

        trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
        train_valid_set = datasets.MNIST(root=root, train=True, transform=trans)
        test_set = datasets.MNIST(root=root, train=False, transform=trans)

        train_valid_set = MNIST_bis(dataset=train_valid_set, size=train_size+valid_size, digits_to_keep=[0,1,2])
        test_set = MNIST_bis(dataset=test_set, size=test_size, digits_to_keep=[0,1,2])

        train_sampler, valid_sampler = train_valid_split(dataset=train_valid_set, train_size=train_size)

        train_loader = DataLoader(dataset=train_valid_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(dataset=train_valid_set, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=True)

    elif dataset == 'mnist_rot':
        root = './data/mnist'
        num_classes = 9

        train_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((26,26)), transforms.ToTensor(), transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
        test_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((26,26)), transforms.RandomRotation((0,360)), transforms.ToTensor(), transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
        train_valid_set = datasets.MNIST(root=root, train=True, transform=train_trans)
        test_set = datasets.MNIST(root=root, train=False, transform=test_trans)

        train_valid_set_bis = MNIST_bis(dataset=train_valid_set, size=train_size+valid_size, digits_to_keep=[0,1,2,3,4,5,6,7,8])
        test_set = MNIST_bis(dataset=test_set, size=test_size, digits_to_keep=[0,1,2,3,4,5,6,7,8])

        train_sampler, valid_sampler = train_valid_split(dataset=train_valid_set_bis, train_size=train_size)

        train_loader = DataLoader(dataset=train_valid_set_bis, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(dataset=train_valid_set_bis, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=True)

    elif dataset == 'mnist_trans':
        root = './data/mnist'
        num_classes = 9

        train_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((26,26)), transforms.ToTensor(), transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
        test_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((26,26)), RandomTranslation(horizontal=6, vertical=6), transforms.ToTensor(), transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
        train_valid_set = datasets.MNIST(root=root, train=True, transform=train_trans)
        test_set = datasets.MNIST(root=root, train=False, transform=test_trans)
        
        train_valid_set_bis = MNIST_bis(dataset=train_valid_set, size=train_size+valid_size, digits_to_keep=[0,1,2,3,4,5,6,7,8])
        test_set = MNIST_bis(dataset=test_set, size=test_size, digits_to_keep=[0,1,2,3,4,5,6,7,8])

        train_sampler, valid_sampler = train_valid_split(dataset=train_valid_set_bis, train_size=train_size)

        train_loader = DataLoader(dataset=train_valid_set_bis, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(dataset=train_valid_set_bis, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, drop_last=True)

    elif dataset == 'eth80':
        root = './data/eth80'
        num_classes = 8

        trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((50,50)), transforms.ToTensor(), transforms.Normalize(mean=ETH80_MEAN, std=ETH80_STD)])
        complete_set = datasets.ImageFolder(root=root, transform=trans)
        class_names = complete_set.classes

        train_sampler, valid_sampler, test_sampler = train_valid_test_split(dataset=complete_set, train_size=train_size, valid_size=valid_size)
        
        train_loader = DataLoader(dataset=complete_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True) 
        valid_loader = DataLoader(dataset=complete_set, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=4, pin_memory=True, drop_last=True) 
        test_loader = DataLoader(dataset=complete_set, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4, pin_memory=True, drop_last=True)

    else:
        raise ValueError('Specified dataset does not exist.')

    logger.debug('Class frequency train loader: {} validation loader: {} test loader: {}'.format(
        count_class_freq(train_loader, num_classes),count_class_freq(valid_loader, num_classes), count_class_freq(test_loader, num_classes))
        )
    logging.info('Loaded {} dataset with the split {}-{}-{} for the [train]-[valid]-[test] setup.'.format(dataset, len(train_loader)*BATCH_SIZE, len(valid_loader)*BATCH_SIZE, len(test_loader)*BATCH_SIZE))


    return train_loader, valid_loader, test_loader, get_dim(train_loader)

