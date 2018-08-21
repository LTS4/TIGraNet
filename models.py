#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Neural Networks models module.
"""

import numpy as np
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from layers import SpectralConv, DynamicPool, Statistic
from utils import init_mask
from configuration import *
from paths import SAVED_DATA, DEBUG_DIR_MNIST_012, DEBUG_DIR_MNIST_rot, DEBUG_DIR_ETH80

logger = logging.getLogger(__name__)

class TIGraNet(nn.Module):
    def __init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights=False, freeze_sc_weights=False):
        super(TIGraNet, self).__init__()
        self.num_nodes = dim**2
        self.laplacian_matrix = laplacian_matrix
        self.shifted_laplacian_matrix = shifted_laplacian_matrix
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mask = init_mask(num_nodes=self.num_nodes, batch_size=self.batch_size)
        self.load_pretrained_weights = load_pretrained_weights
        self.freeze_sc_weights = freeze_sc_weights

        self.loss_function = torch.nn.CrossEntropyLoss()

    def init_pretrained_weights(self, name):
        """Initialize the weights of the model with pretrained weights."""

        self.spectral_conv1.alpha.weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'alpha_0.npy'))))
        self.spectral_conv1.beta.weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'beta_0.npy'))).unsqueeze(0))
        self.spectral_conv2.alpha.weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'alpha_1.npy'))))
        self.spectral_conv2.beta.weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'beta_1.npy'))).unsqueeze(0))
        self.fully_connected[0].weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'W_1.npy'))).t())
        self.fully_connected[0].bias = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'b_1.npy'))))
        self.fully_connected[2].weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'W_2.npy'))).t())
        self.fully_connected[2].bias = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'b_2.npy'))))
        self.fully_connected[4].weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'W_3.npy'))).t())
        self.fully_connected[4].bias = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'b_3.npy'))))
        self.fully_connected[6].weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'W_last.npy'))).t())
        self.fully_connected[6].bias = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'b_last.npy'))))
        
        if name=='mnist_012':
            self.spectral_conv3.alpha.weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'alpha_2.npy'))))
            self.spectral_conv3.beta.weight = nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, name, 'parameters', 'beta_2.npy'))).unsqueeze(0))

    def prepare_input(self, input):
        input = input.view(self.batch_size, 1, self.num_nodes)
        input = input - torch.mean(input, 2, True)
        input = input.transpose(1,2)
        return input
    
    def step(self, input, target, train):
        if train:
            self.train()
        else:
            self.eval()
        self.optimizer.zero_grad()
        out = self.forward(input)
        loss = self.loss_function(out, target)

        if train:
            loss.backward()

        self.optimizer.step()

        return loss.item()

    def predict(self, input):
        self.eval()
        output = self.forward(input)
        _, output = torch.max(output, 1)

        return output

class TIGraNet_mnist_012(TIGraNet):
    def __init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights=False, freeze_sc_weights=False):
        TIGraNet.__init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights, freeze_sc_weights)

        # Main layers
        self.spectral_conv1 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=1,
            filter_size_out=3,
            degree_of_polynomial=4,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool1 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=3,
            num_active_nodes=200,
            mask=self.mask
            )
        self.spectral_conv2 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=3,
            filter_size_out=6,
            degree_of_polynomial=4,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool2 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=6,
            num_active_nodes=100,
            mask=self.mask
            )
        self.spectral_conv3 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=6,
            filter_size_out=9,
            degree_of_polynomial=4,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool3 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=9,
            num_active_nodes=50,
            mask=self.mask
            )
        self.statistic = Statistic(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=9,
            degree_of_polynomial=9,
            shifted_laplacian_matrix=self.shifted_laplacian_matrix
            )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=180, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=80),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=80, out_features= 60),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=60, out_features=3)
        )
        
        if load_pretrained_weights:
            self.init_pretrained_weights(name='mnist_012')

            # random checks
            assert (self.spectral_conv2.alpha.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_012', 'parameters', 'alpha_1.npy'))))).all()
            assert (self.spectral_conv2.beta.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_012', 'parameters', 'beta_1.npy'))).unsqueeze(0))).all()
            assert (self.fully_connected[2].weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_012', 'parameters', 'W_2.npy'))).t())).all()

            logger.info('Loaded pretrained weights.')
        else:
            self.init_weights_default()
            logger.info('Loaded weights using uniform distribution in [0,1].')

        if freeze_sc_weights:
            # freeze the parameters of the spectral conv layer
            for m in self.modules():
                if isinstance(m, SpectralConv):
                    m.alpha.weight.requires_grad = False
                    m.beta.weight.requires_grad = False

            logger.info('Freezed spectral conv weights.')

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)

        logger.info('Loaded {} optimizer.'.format(type(self.optimizer).__name__))


    def init_weights_default(self):
        """Initialize the weights of the model with uniform distribution in [0,1]."""

        for m in self.modules():
            if isinstance(m, SpectralConv):
                nn.init.uniform_(m.alpha.weight)
                nn.init.uniform_(m.beta.weight)

    def forward(self, input):
        prepared_input = self.prepare_input(input)
        filter_operator1, y1, spectral_conv1 = self.spectral_conv1(prepared_input, self.mask)
        mask1, dynamic_pool1 = self.dynamic_pool1(spectral_conv1, self.mask)
        filter_operator2, y2, spectral_conv2 = self.spectral_conv2(spectral_conv1, mask1)
        mask2, dynamic_pool2 = self.dynamic_pool2(spectral_conv2, mask1)
        filter_operator3, y3, spectral_conv3 = self.spectral_conv3(spectral_conv2, mask2)
        mask3, dynamic_pool3 = self.dynamic_pool3(spectral_conv3, mask2)
        statistic = self.statistic(spectral_conv3, mask3)
        output = self.fully_connected(statistic)

        if GENERATE_SAVE:
            # save all intermediary steps for debugging
            variables = [prepared_input, filter_operator1, y1, spectral_conv1, filter_operator2, y2, spectral_conv2, filter_operator3, y3, spectral_conv3, mask1, mask2, mask3, statistic, output]
            variables_names = ['prepared_input', 'filter_operator1', 'y1', 'spectral_conv1', 'filter_operator2', 'y2', 'spectral_conv2', 'filter_operator3', 'y3', 'spectral_conv3', 'mask1', 'mask2', 'mask3', 'statistic', 'output']
            tuples = zip(variables, variables_names)

            for v, n in tuples:
                # np.save(DEBUG_DIR_MNIST_012 + 'constant_weights/' + n + '_p', v.detach().numpy())
                np.save(DEBUG_DIR_MNIST_012 + 'pretrained_weights/' + n + '_p_pw', v.detach().numpy())

        return output

class TIGraNet_mnist_rot(TIGraNet):
    def __init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights=False, freeze_sc_weights=False):
        TIGraNet.__init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights, freeze_sc_weights)

        # Main layers
        self.spectral_conv1 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=1,
            filter_size_out=10,
            degree_of_polynomial=4,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool1 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=10,
            num_active_nodes=600,
            mask=self.mask
            )
        self.spectral_conv2 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=10,
            filter_size_out=20,
            degree_of_polynomial=4,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool2 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=20,
            num_active_nodes=300,
            mask=self.mask
            )
        self.statistic = Statistic(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=20,
            degree_of_polynomial=13,
            shifted_laplacian_matrix=self.shifted_laplacian_matrix
            )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=560, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=300),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features= 100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=9)
        )
        
        if load_pretrained_weights:
            self.init_pretrained_weights(name='mnist_rot')

            # random checks
            assert (self.spectral_conv2.alpha.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_rot', 'parameters', 'alpha_1.npy'))))).all()
            assert (self.spectral_conv2.beta.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_rot', 'parameters', 'beta_1.npy'))).unsqueeze(0))).all()
            assert (self.fully_connected[2].weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_rot', 'parameters', 'W_2.npy'))).t())).all()

            logger.info('Loaded pretrained weights.')

        if freeze_sc_weights:
            # freeze the parameters of the spectral conv layer
            for m in self.modules():
                if isinstance(m, SpectralConv):
                    m.alpha.weight.requires_grad = False
                    m.beta.weight.requires_grad = False

            logger.info('Freezed spectral conv weights.')

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        logger.info('Loaded {} optimizer.'.format(type(self.optimizer).__name__))

    def forward(self, input):
        prepared_input = self.prepare_input(input)
        filter_operator1, y1, spectral_conv1 = self.spectral_conv1(prepared_input, self.mask)
        mask1, dynamic_pool1 = self.dynamic_pool1(spectral_conv1, self.mask)
        filter_operator2, y2, spectral_conv2 = self.spectral_conv2(spectral_conv1, mask1)
        mask2, dynamic_pool2 = self.dynamic_pool2(spectral_conv2, mask1)
        statistic = self.statistic(spectral_conv2, mask2)
        output = self.fully_connected(statistic)

        if GENERATE_SAVE:
            # save all intermediary steps for debugging
            variables = [prepared_input, filter_operator1, y1, spectral_conv1, filter_operator2, y2, spectral_conv2, mask1, mask2, statistic, output]
            variables_names = ['prepared_input', 'filter_operator1', 'y1', 'spectral_conv1', 'filter_operator2', 'y2', 'spectral_conv2', 'mask1', 'mask2', 'statistic', 'output']
            tuples = zip(variables, variables_names)

            for v, n in tuples:
                np.save(DEBUG_DIR_MNIST_rot + n + '_p_pw', v.detach().numpy())

        return output

class TIGraNet_mnist_trans(TIGraNet):
    def __init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights=False, freeze_sc_weights=False):
        TIGraNet.__init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights, freeze_sc_weights)

        # Main layers
        self.spectral_conv1 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=1,
            filter_size_out=10,
            degree_of_polynomial=7,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool1 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=10,
            num_active_nodes=600,
            mask=self.mask
            )
        self.spectral_conv2 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=10,
            filter_size_out=20,
            degree_of_polynomial=7,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool2 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=20,
            num_active_nodes=300,
            mask=self.mask
            )
        self.statistic = Statistic(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=20,
            degree_of_polynomial=11,
            shifted_laplacian_matrix=self.shifted_laplacian_matrix
            )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=480, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=300),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features= 100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=9)
        )
        
        if load_pretrained_weights:
            self.init_pretrained_weights(name='mnist_trans')

            # random checks
            assert (self.spectral_conv2.alpha.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_trans', 'parameters', 'alpha_1.npy'))))).all()
            assert (self.spectral_conv2.beta.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_trans', 'parameters', 'beta_1.npy'))).unsqueeze(0))).all()
            assert (self.fully_connected[2].weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'mnist_trans', 'parameters', 'W_2.npy'))).t())).all()

            logger.info('Loaded pretrained weights.')

        if freeze_sc_weights:
            # freeze the parameters of the spectral conv layer
            for m in self.modules():
                if isinstance(m, SpectralConv):
                    m.alpha.weight.requires_grad = False
                    m.beta.weight.requires_grad = False

            logger.info('Freezed spectral conv weights.')

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        logger.info('Loaded {} optimizer.'.format(type(self.optimizer).__name__))

    def forward(self, input):
        prepared_input = self.prepare_input(input)
        filter_operator1, y1, spectral_conv1 = self.spectral_conv1(prepared_input, self.mask)
        mask1, dynamic_pool1 = self.dynamic_pool1(spectral_conv1, self.mask)
        filter_operator2, y2, spectral_conv2 = self.spectral_conv2(spectral_conv1, mask1)
        mask2, dynamic_pool2 = self.dynamic_pool2(spectral_conv2, mask1)
        statistic = self.statistic(spectral_conv2, mask2)
        output = self.fully_connected(statistic)

        return output

class TIGraNet_eth80(TIGraNet):
    def __init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights=False, freeze_sc_weights=False):
        TIGraNet.__init__(self, dim, laplacian_matrix, shifted_laplacian_matrix, batch_size, learning_rate, load_pretrained_weights, freeze_sc_weights)

        # Main layers
        self.spectral_conv1 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=1,
            filter_size_out=10,
            degree_of_polynomial=5,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool1 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=10,
            num_active_nodes=600,
            mask=self.mask
            )
        self.spectral_conv2 = SpectralConv(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            filter_size_in=10,
            filter_size_out=20,
            degree_of_polynomial=5,
            laplacian_matrix=self.laplacian_matrix,
            mask=self.mask
            )
        self.dynamic_pool2 = DynamicPool(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=20,
            num_active_nodes=300,
            mask=self.mask
            )
        self.statistic = Statistic(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            num_filters=20,
            degree_of_polynomial=11,
            shifted_laplacian_matrix=self.shifted_laplacian_matrix
            )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=480, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=300),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features= 100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=8)
        )
        
        if load_pretrained_weights:
            self.init_pretrained_weights(name='eth80')

            # random checks
            assert (self.spectral_conv2.alpha.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'eth80', 'parameters', 'alpha_1.npy'))))).all()
            assert (self.spectral_conv2.beta.weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'eth80', 'parameters', 'beta_1.npy'))).unsqueeze(0))).all()
            assert (self.fully_connected[2].weight == nn.Parameter(torch.from_numpy(np.load(os.path.join(SAVED_DATA, 'eth80', 'parameters', 'W_2.npy'))).t())).all()

            logger.info('Loaded pretrained weights.')

        if freeze_sc_weights:
            # freeze the parameters of the spectral conv layer
            for m in self.modules():
                if isinstance(m, SpectralConv):
                    m.alpha.weight.requires_grad = False
                    m.beta.weight.requires_grad = False

            logger.info('Freezed spectral conv weights.')

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        logger.info('Loaded {} optimizer.'.format(type(self.optimizer).__name__))

    def forward(self, input):
        prepared_input = self.prepare_input(input)
        filter_operator1, y1, spectral_conv1 = self.spectral_conv1(prepared_input, self.mask)
        mask1, dynamic_pool1 = self.dynamic_pool1(spectral_conv1, self.mask)
        filter_operator2, y2, spectral_conv2 = self.spectral_conv2(spectral_conv1, mask1)
        mask2, dynamic_pool2 = self.dynamic_pool2(spectral_conv2, mask1)
        statistic = self.statistic(spectral_conv2, mask2)
        output = self.fully_connected(statistic)

        if GENERATE_SAVE:
            # save all intermediary steps for debugging
            variables = [prepared_input, filter_operator1, y1, spectral_conv1, filter_operator2, y2, spectral_conv2, mask1, mask2, statistic, output]
            variables_names = ['prepared_input', 'filter_operator1', 'y1', 'spectral_conv1', 'filter_operator2', 'y2', 'spectral_conv2', 'mask1', 'mask2', 'statistic', 'output']
            tuples = zip(variables, variables_names)

            for v, n in tuples:
                np.save(DEBUG_DIR_ETH80 + n + '_p_pw', v.detach().numpy())

        return output