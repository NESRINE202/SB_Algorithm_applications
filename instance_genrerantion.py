import numpy as np
import random as rd

def generate_instance(size):
       M = np.random.uniform(low=0, high=10, size=(size, size))
       H = np.zeros(size)

       return M, H