#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Data loader for the PyTorch framework.
"""

from tqdm import tqdm
import os, re

import torch
import torch.utils.data as data

from utils import select

class MNIST_bis(data.Dataset):
    def __init__(self, dataset, size, digits_to_keep, stratified_sampling=True):
        self.dataset=dataset
        self.indices=select(dataset, size, digits_to_keep, stratified_sampling)
    
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]