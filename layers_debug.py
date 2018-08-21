#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Layer debugging module.
"""

import logging

from saved_datasets import load_saved_dataset
from configuration import BATCH_SIZE, LEARNING_RATE
from models import TIGraNet_mnist_012, TIGraNet_mnist_rot, TIGraNet_eth80
from debug import display_weights, init_weights_constant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_, _, test_loader, dim, laplacian_matrix, shifted_laplacian_matrix = load_saved_dataset(name='mnist_rot')

# create model and initialize weights
model = TIGraNet_mnist_rot(
    dim=dim,
    laplacian_matrix=laplacian_matrix,
    shifted_laplacian_matrix=shifted_laplacian_matrix,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    load_pretrained_weights=True
    )

# init_weights_constant(model)
display_weights(model)

for data, target in test_loader:
    model.forward(data)
    break