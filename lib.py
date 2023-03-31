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
