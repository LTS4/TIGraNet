#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Layers module.
"""

import logging
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from graph import compute_laplacian_power_basis, compute_chebyshev_polynomials, compute_statistics
from configuration import *

logger = logging.getLogger(__name__)

class SpectralConv(nn.Module):
    def __init__(self, batch_size, num_nodes, filter_size_in, filter_size_out, degree_of_polynomial, laplacian_matrix, mask):
        super(SpectralConv, self).__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.filter_size_in = filter_size_in
        self.filter_size_out = filter_size_out
        self.degree_of_polynomial = degree_of_polynomial
        self.laplacian_matrix = laplacian_matrix
        self.mask = mask # B x N x 1
        self.alpha = nn.Linear(self.degree_of_polynomial, self.filter_size_out, bias=False)
        self.beta = nn.Linear(self.filter_size_in, 1, bias=False)
        self.laplacian_tensor = compute_laplacian_power_basis(self.laplacian_matrix, self.degree_of_polynomial).to(DEVICE)
        
    def apply_filter_operator(self, input, filter_operator):
        """Apply the filter operator for each input channel."""

        input = input.permute(0,2,1).contiguous().view(-1,self.num_nodes)
        filter_operator = filter_operator.view(self.num_nodes, -1)
        output = torch.matmul(input, filter_operator).view(self.batch_size, self.filter_size_in, self.num_nodes, self.filter_size_out).permute(0,2,3,1)

        matched_mask = self.mask.unsqueeze(2).repeat(1,1,self.filter_size_out,1)
        output = output * matched_mask

        # Debug
        logger.debug('Filter operator with matched dimensions of spectral conv layer: {}'.format(filter_operator.shape))
        logger.debug('Output after applying filter operator on input of spectral conv layer: {}'.format(output.size()))

        return output

    def forward(self, input, mask):

        self.mask = mask
        filter_operator = self.alpha(self.laplacian_tensor) # N x N x OUT
        y = self.apply_filter_operator(input, filter_operator) # B x OUT x N x IN
        z = self.beta(y).squeeze()
        
        # Debug
        logger.debug('Input of spectral conv layer: {}'.format(input.shape))
        logger.debug('Laplacian tensor of spectral conv layer: {}'.format(self.laplacian_tensor.shape))
        logger.debug('Filter operator of spectral conv layer: {}'.format(filter_operator.shape))
        logger.debug('Y of spectral conv layer: {}'.format(y.shape))
        logger.debug('Z of spectral conv layer: {}'.format(z.shape))

        return filter_operator, y, z

class DynamicPool(nn.Module):
    def __init__(self, batch_size, num_nodes, num_filters, num_active_nodes, mask):
        super(DynamicPool, self).__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_filters = num_filters
        self.num_active_nodes = num_active_nodes
        self.mask = mask
        self.epsilon = 10**(-10)

    def get_max_indices(self, input):
        """Return the maximum indices from the input based on the mask values."""
        
        min_element = torch.min(torch.abs(input.contiguous().view(-1)))
        input_temp = input + min_element + self.epsilon
        masked_input_temp = input_temp * self.mask
        values, indices = torch.sort(masked_input_temp, dim=1, descending=True)

        return indices[:, :self.num_active_nodes,:]

    def update_mask(self, indices):
        """Update the mask with the corresponding indices."""

        indices = indices.view(self.batch_size, -1)
        updated_mask = torch.zeros_like(self.mask.squeeze(-1)).scatter_(1, indices, 1)

        return updated_mask.unsqueeze(-1)

    def forward(self, input, mask):
        max_indices = self.get_max_indices(input)
        updated_mask = self.update_mask(max_indices)
        masked_input = input * updated_mask

        return updated_mask, masked_input

class Statistic(nn.Module):
    def __init__(self, batch_size, num_nodes, num_filters, degree_of_polynomial, shifted_laplacian_matrix):
        super(Statistic, self).__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_filters = num_filters
        self.degree_of_polynomial = degree_of_polynomial
        self.shifted_laplacian_matrix = shifted_laplacian_matrix

    def forward(self, input, mask):
        chebyshev_polynomials = compute_chebyshev_polynomials(
            input=input.contiguous(),
            num_nodes=self.num_nodes,
            num_filters=self.num_filters, 
            shifted_laplacian_matrix=self.shifted_laplacian_matrix,
            degree_of_polynomial=self.degree_of_polynomial
            )
        feature_vector = compute_statistics(
            chebyshev_polynomials=chebyshev_polynomials,
            num_filters=self.num_filters,
            degree_of_polynomial=self.degree_of_polynomial,
            mask=mask
            )

        return feature_vector.view(self.batch_size, -1)

