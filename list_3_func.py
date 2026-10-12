import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
from decimal import Decimal

def box_muller(size, mean=2, variance=2):
    u1 = np.random.uniform(0,1,size=size//2)
    u2 = np.random.uniform(0,1,size=size//2)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2) * np.sqrt(variance) + mean
    z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2) * np.sqrt(variance) + mean
    
    return np.concatenate([z0, z1])

