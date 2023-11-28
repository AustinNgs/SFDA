import torch
import numpy as np
import sys

sys.path[0] = '/home/lab-wu.shibin/SFDA'

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def euclidean_dist(x, y=None):
    """
    Compute all pairwise distances between vectors in X and Y matrices.
    :param x: numpy array, with size of (d, m)
    :param y: numpy array, with size of (d, n)
    :return: EDM:   numpy array, with size of (m,n). 
                    Each entry in EDM_{i,j} represents the distance between row i in x and row j in y.
    """
    if y is None:
        y = x

    # calculate Gram matrices
    G_x = np.matmul(x.T, x)
    G_y = np.matmul(y.T, y)

    # convert diagonal matrix into column vector
    diag_Gx = np.reshape(np.diag(G_x), (-1, 1))
    diag_Gy = np.reshape(np.diag(G_y), (-1, 1))

    # Compute Euclidean distance matrix
    EDM = diag_Gx + diag_Gy.T - 2*np.matmul(x.T, y)  # broadcasting
    d = np.sqrt(EDM)
    return d.sum()
    
