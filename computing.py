from ising_model import IsingModel
import instance_genrerantion
import numpy as np
import datetime

# This file is for doing the actual simulation and saving the results to be latter annalysed

# compute for a sigle instance
def compute_single_instance(instance_size, step, n_itterations, n_cond_init, J, H, savetofile=True):
    # run the algorithm
    ising_model = IsingModel(step, n_itterations, n_cond_init, J, H)
    states, energies = ising_model.simulate()

    # Save all the data (parameters, instance, results) inside of a file
    # Contains the parameters, the instance matrix and the states and enrgies matrices
    path = ''
    if savetofile:
        now = datetime.datetime.now()
        datestr = now.strftime("%d%m%Y-%H%M")
        filename = f"{instance_size}_{step}_{n_itterations}_{n_cond_init}_{datestr}.npz"
        path = f'computing_results/{filename}'
        np.savez(path, J=J, H=H, states=states, energies=energies)

    return states, energies, path

if __name__ == '__main__':
    compute_single_instance(instance_size=20, step=0.001, n_itterations=10000, n_cond_init=20)