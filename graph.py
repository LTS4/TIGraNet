#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Graph signal processing functions.
"""

import numpy as np
import logging

import torch
from torch.autograd import Variable

from configuration import BATCH_SIZE, DEVICE

logger = logging.getLogger(__name__)

def compute_adjacency_matrix(dim, k=4):
    """Create a k nearest neighbors adjacency matrix from a (dim, dim) matrix."""
    
    logger.debug('Creating adjacency matrix with {}NN version.'.format(k))

    def get_chebyshev_indices(dim, x, y, k=4, radius=1):
        """Return the indices away from (x,y) by given radius in the Chebyshev distance metric for a square matrix of size (dim, dim)."""
        
        l = []
        lowerX = np.maximum(0, x - radius)
        upperX = np.minimum(dim - 1, x + radius)
        lowerY = np.maximum(0, y - radius)
        upperY = np.minimum(dim - 1, y + radius)

        if k == 4:
            for i in range(lowerX, upperX+1):
                if not i==x:
                    l.extend([i*dim + y]) 
            for j in range(lowerY, upperY+1):
                if not j==y:
                    l.extend([x*dim + j])
        elif k == 8:
            for i in range(lowerX, upperX+1):
                for j in range(lowerY, upperY+1):
                    if not (i==x and j==y):
                        l.extend([i*dim + j])            
        else:
            raise ValueError('Specified KNN version for adjacency matrix is not defined: currently 4NN and 8NN are supported.')          
        
        return l, len(l)

    size = dim**2

    i1 = []
    i2 = []
    indices = []
    values = []

    for j in range(size):
        x = j//dim
        y = j%dim
        i, l = get_chebyshev_indices(dim, x, y, k)
        i1.extend([j] * l)
        i2.extend(i)
        values.extend(list(np.ones(l)))
    
    indices = torch.LongTensor([i1, i2])
    values = torch.FloatTensor(values)
    sparse_adjacency_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([size, size]))

    return sparse_adjacency_matrix

def to_sparse(x):
    """Convert dense tensor to sparse format."""

    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]

    return sparse_tensortype(indices, values, x.size())

def compute_normalized_laplacian(sparse_adjacency_matrix):
    """Return the Laplacian of the graph G from the adjacency matrix."""

    size = sparse_adjacency_matrix.size()[0]
    adjacency_matrix = sparse_adjacency_matrix.to_dense()
    degree = torch.sum(adjacency_matrix, dim=1)
    degree = 1 / np.sqrt(degree)
    degree_matrix = torch.diag(degree)
    idendity_matrix = torch.diag(torch.ones(size))
    laplacian_matrix = idendity_matrix - torch.mm(torch.mm(degree_matrix,adjacency_matrix),degree_matrix)

    return laplacian_matrix

def compute_laplacian_power_basis(laplacian_matrix, degree_of_polynomial):
    """Return a tensor containing the Laplacian matrices of power 0 to degree_of_polynomial."""
    
    #laplacian_matrix = laplacian_matrix.to_dense() # because sparseTensor has no attribute unsqueeze used in .stack()
    # convert it to numpy matrix to benefit from the matrix power operation (**)
    laplacian_power_basis = [torch.from_numpy(np.matrix(laplacian_matrix.numpy())**d) for d in range(degree_of_polynomial)]
    laplacian_tensor = torch.stack(laplacian_power_basis)
    laplacian_tensor = laplacian_tensor.transpose(0,2) # N x N x M

    #return to_sparse(laplacian_tensor)
    return laplacian_tensor

def shift_laplacian(laplacian_matrix, dim):
    """Shift the Laplacian eigenvalues to [-1, 1]."""
    
    #laplacian_matrix = laplacian_matrix/2
    idendity_matrix = torch.eye(dim**2)
    shifted_laplacian_matrix = laplacian_matrix - idendity_matrix

    return shifted_laplacian_matrix

def compute_laplacians(dim):
    """Compute the normalized and shifted Laplacian matrices for a specific image dimension."""

    adjacency_matrix = compute_adjacency_matrix(dim=dim)
    laplacian_matrix = compute_normalized_laplacian(sparse_adjacency_matrix=adjacency_matrix)
    shifted_laplacian_matrix = shift_laplacian(laplacian_matrix, dim).to(DEVICE)
    
    return laplacian_matrix, shifted_laplacian_matrix

def compute_chebyshev_polynomials(input, num_nodes, num_filters, shifted_laplacian_matrix, degree_of_polynomial):
    """Return the Chebyshev polynomials of order up to degree_of_polynomial."""

    chebyshev_polynomials = torch.FloatTensor(BATCH_SIZE, num_filters, num_nodes, degree_of_polynomial+1).zero_().to(DEVICE)
    
    t_0 = input.transpose(1,2) # B x F x N
    chebyshev_polynomials[:,:,:,0] = t_0
    t_1 = torch.matmul(t_0, shifted_laplacian_matrix)
    chebyshev_polynomials[:,:,:,1] = t_1

    # Debug
    logging.debug('Input of statistical layer: {}'.format(input.shape))
    logging.debug('Shifted Laplacian matrix of statistical layer: {}'.format(shifted_laplacian_matrix.shape))
    logging.debug('t_0 of Chebyshev polynomial: {}'.format(t_0.shape))
    logging.debug('t_1 of Chebyshev polynomial: {}'.format(t_1.shape))
    
    for k in range(2, degree_of_polynomial+1):
        chebyshev_polynomials[:,:,:,k] = 2*torch.matmul(chebyshev_polynomials[:,:,:,k-1].clone(), shifted_laplacian_matrix) - chebyshev_polynomials[:,:,:,k-2].clone()            
    
    return chebyshev_polynomials

def compute_statistics(chebyshev_polynomials, num_filters, degree_of_polynomial, mask):
    """Return the mean and the standard deviation of the Chebyshev polynomials."""

    abs_chebyshev_polynomials = torch.abs(chebyshev_polynomials * mask.unsqueeze(1))
    mean = torch.mean(abs_chebyshev_polynomials, dim=2)
    std = torch.std(abs_chebyshev_polynomials, dim=2)

    # to match the Theano implementation ordering in order to be able to use pretrained weights from it /!\
    feature_vector = torch.cat((mean, std), dim=1).transpose(1,2).contiguous()
    
    return feature_vector
