from ising_modelV1 import IsingModel
import instance_genrerantionV2
import numpy as np
import datetime

# This file is for doing the actual simulation and saving the results to be latter annalysed

# compute for a sigle instance
def compute_single_instance(instance_size, step, n_itterations, n_cond_init, J, H,forces, temperature=None, a=None):
    # run the algorithm
    ising_model = IsingModel(step, n_itterations, n_cond_init, J, H,forces, temperature, a)
    states = ising_model.simulate()


    return states