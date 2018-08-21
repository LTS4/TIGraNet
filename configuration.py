#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Configuration file.
"""
import torch

# settings for the implementation
SEED = 7

# used device (CPU vs GPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# settings for the dataset
DATASET = 'mnist_012'
TRAIN_SIZE = 500
VALID_SIZE = 100
TEST_SIZE = 100

# properties of the model
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
USE_PRETRAINED_MODEL = False

# settings of the run
TRAIN = True
PATIENCE = 20

# debugging
GENERATE_SAVE = False