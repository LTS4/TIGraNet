#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Saved datasets module.
"""

import numpy as np
import logging
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

from paths import SAVED_DATA
from configuration import *
from graph import shift_laplacian
from utils import get_dim, count_class_freq

logger = logging.getLogger(__name__)

def load_saved_dataset(name, data_path=SAVED_DATA):
    """Load the saved data."""

    train_data = torch.from_numpy(np.load(os.path.join(data_path, name, 'dataset', name + '_train_signals.npy'))).float()
    valid_data = torch.from_numpy(np.load(os.path.join(data_path, name, 'dataset', name + '_val_signals.npy'))).float()
    test_data = torch.from_numpy(np.load(os.path.join(data_path, name, 'dataset', name + '_test_signals.npy'))).float()

    train_labels = torch.from_numpy(np.load(os.path.join(data_path, name, 'dataset', name + '_train_labels.npy'))).long()
    valid_labels = torch.from_numpy(np.load(os.path.join(data_path, name, 'dataset', name + '_val_labels.npy'))).long()
    test_labels = torch.from_numpy(np.load(os.path.join(data_path, name, 'dataset', name + '_test_labels.npy'))).long()

    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    dim = int(np.sqrt(get_dim(train_data)))

    laplacian_matrix = torch.from_numpy(np.load(os.path.join(data_path, name, 'dataset', name + '_laplacian.npy'))).float()
    shifted_laplacian_matrix = shift_laplacian(laplacian_matrix, dim).to(DEVICE)

    if name == 'mnist_012':
        num_classes = 3
    elif name == 'eth80':
        num_classes = 8
    else:
        num_classes = 9

    logger.info('Class frequency \ntrain loader: {} \nvalidation loader: {} \ntest loader: {}'.format(
        count_class_freq(train_loader, num_classes),count_class_freq(valid_loader, num_classes), count_class_freq(test_loader, num_classes))
        )
    logging.info('Loaded saved {} dataset with the split {}-{}-{} for the [train]-[valid]-[test] setup.'.format(name, len(train_loader)*BATCH_SIZE, len(valid_loader)*BATCH_SIZE, len(test_loader)*BATCH_SIZE))

    return train_loader, valid_loader, test_loader, dim, laplacian_matrix, shifted_laplacian_matrix




