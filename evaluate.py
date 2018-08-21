#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Test module.
"""

import numpy as np
import logging
from tqdm import tqdm
import sys

from saved_datasets import load_saved_dataset
from graph import compute_laplacians 
from utils import load_pretrained_model, snapshot
from paths import SAVED_MODELS_DIR
from configuration import *
from models import TIGraNet_mnist_012, TIGraNet_mnist_rot, TIGraNet_mnist_trans, TIGraNet_eth80

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
            load_pretrained_weights=True
            )
    elif dataset_name == 'mnist_rot':
        model = TIGraNet_mnist_rot(
            dim=dim,
            laplacian_matrix=laplacian_matrix,
            shifted_laplacian_matrix=shifted_laplacian_matrix,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            load_pretrained_weights=True
            )
    elif dataset_name == 'mnist_trans':
        model = TIGraNet_mnist_trans(
            dim=dim,
            laplacian_matrix=laplacian_matrix,
            shifted_laplacian_matrix=shifted_laplacian_matrix,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            load_pretrained_weights=True
            )
    elif dataset_name == 'eth80':
        model = TIGraNet_eth80(
            dim=dim,
            laplacian_matrix=laplacian_matrix,
            shifted_laplacian_matrix=shifted_laplacian_matrix,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            load_pretrained_weights=True
            )
    
    return model

# get arguments from command line
if len(sys.argv) != 2:
    print('Usage: python3 evaluate.py [DATASET]')
    sys.exit(1)
else:
    dataset_name = sys.argv[-1]
    if dataset_name not in ['mnist_012', 'mnist_rot', 'mnist_trans', 'eth80']:
        print('DATASET available: mnist_012, mnist_rot, mnist_trans or eth80')
        sys.exit(1)

# prepare data and model
_, _, test_loader, dim, laplacian_matrix, shifted_laplacian_matrix = load_saved_dataset(name=dataset_name)
model = load_model(dataset_name=dataset_name, dim=dim, laplacian_matrix=laplacian_matrix, shifted_laplacian_matrix=shifted_laplacian_matrix)

# pass it to GPU if available
model.to(DEVICE)

# evaluate on testing set
logging.info('Testing...')
acc_valid = 0
test_samples_size = len(test_loader) * BATCH_SIZE
for data, target in tqdm(test_loader):
    
    data = data.to(DEVICE)
    y_pred = model.predict(data)
    acc_valid += torch.eq(y_pred.cpu(),target).sum().item()

error_test = 100 - 100 * acc_valid / test_samples_size
print('test error: {:.2f} %'.format(error_test))