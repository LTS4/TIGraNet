#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Main module.
"""

import numpy as np
import datetime
import logging
from tqdm import tqdm
import time
import sys

from torch.autograd import Variable

from datasets import load_dataset
from saved_datasets import load_saved_dataset
from graph import compute_laplacians 
from utils import snapshot, load_pretrained_model
from plot import plot_loss, plot_error
from paths import SAVED_MODELS_DIR
from configuration import *
from models import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(dataset_name, dim, laplacian_matrix, shifted_laplacian_matrix):
    """Load the model associated with the dataset."""

    if dataset_name == 'mnist_012':
        model = TIGraNet_mnist_012(
            dim=dim,
            laplacian_matrix=laplacian_matrix,
            shifted_laplacian_matrix=shifted_laplacian_matrix,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            freeze_sc_weights=True
            )
    elif dataset_name == 'mnist_rot':
        model = TIGraNet_mnist_rot(
            dim=dim,
            laplacian_matrix=laplacian_matrix,
            shifted_laplacian_matrix=shifted_laplacian_matrix,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            freeze_sc_weights=True
            )
    elif dataset_name == 'mnist_trans':
        model = TIGraNet_mnist_trans(
            dim=dim,
            laplacian_matrix=laplacian_matrix,
            shifted_laplacian_matrix=shifted_laplacian_matrix,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            freeze_sc_weights=True
            )
    elif dataset_name == 'eth80':
        model = TIGraNet_eth80(
            dim=dim,
            laplacian_matrix=laplacian_matrix,
            shifted_laplacian_matrix=shifted_laplacian_matrix,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            freeze_sc_weights=True
            )
    
    return model

# get arguments from command line
if len(sys.argv) != 2:
    print('Usage: python3 train.py [DATASET]')
    sys.exit(1)
else:
    dataset_name = sys.argv[-1]
    if dataset_name not in ['mnist_012', 'mnist_rot', 'mnist_trans', 'eth80']:
        print('DATASET available: mnist_012, mnist_rot, mnist_trans or eth80')
        sys.exit(1)

# prepare data and model
train_loader, valid_loader, _, dim, laplacian_matrix, shifted_laplacian_matrix = load_saved_dataset(name=dataset_name)
model = load_model(dataset_name=dataset_name, dim=dim, laplacian_matrix=laplacian_matrix, shifted_laplacian_matrix=shifted_laplacian_matrix)

# pass it to GPU if available
model.to(DEVICE)

logging.info('Training...')
RUN_TIME = '{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now())
RUN_NAME = '{}_{}_{}_{:.0e}'.format(
    type(model).__name__,
    type(model.optimizer).__name__,
    #'F' if model.freeze_sc_weights else 'NF',
    BATCH_SIZE,
    LEARNING_RATE
)
epoch = 0
best_error = (0,100)
loss_history = []
error_history = []
while True:

    # train the model
    loss_train = 0
    acc_train = 0
    for data, target in tqdm(train_loader, desc='Training', leave=False):
        
        data, target = data.to(DEVICE), target.to(DEVICE)
        loss = model.step(data, target, train=True)
        loss_train += loss

        y_pred = model.predict(data)
        acc_train += torch.eq(y_pred.cpu(),target.cpu()).sum().item()
        

    # validate the model
    loss_valid = 0
    acc_valid = 0
    for data, target in tqdm(valid_loader, desc='Validation', leave=False):

        data, target = data.to(DEVICE), target.to(DEVICE)
        loss = model.step(data, target, train=False)
        loss_valid += loss

        y_pred = model.predict(data)
        acc_valid += torch.eq(y_pred.cpu(),target.cpu()).sum().item()

    # print some metrics
    train_samples_size = len(train_loader) * BATCH_SIZE
    valid_samples_size = len(valid_loader) * BATCH_SIZE
    loss_train_epoch = loss_train / train_samples_size
    loss_valid_epoch = loss_valid / valid_samples_size
    error_train_epoch = 100 - 100 * (acc_train / train_samples_size)
    error_valid_epoch = 100 - 100 * (acc_valid / valid_samples_size)
    error_history.append((error_train_epoch, error_valid_epoch))
    loss_history.append((loss_train_epoch, loss_valid_epoch))
    print('Epoch: {} train loss: {:.5f} valid loss: {:.5f} train error: {:.2f} % valid error: {:.2f} %'.format(epoch, loss_train_epoch, loss_valid_epoch, error_train_epoch, error_valid_epoch))

    # check if model is better
    if error_valid_epoch < best_error[1]:
        best_error = (epoch, error_valid_epoch)
        snapshot(SAVED_MODELS_DIR, RUN_TIME, RUN_NAME, True, epoch, error_valid_epoch, model.state_dict(), model.optimizer.state_dict())

    # check that the model is not doing worst over the time
    if best_error[0] + PATIENCE < epoch :
        print('Overfitting. Stopped at epoch {}.' .format(epoch))
        break
    epoch += 1

    plot_loss(RUN_TIME, RUN_NAME, loss_history)
    plot_error(RUN_TIME, RUN_NAME, error_history)
