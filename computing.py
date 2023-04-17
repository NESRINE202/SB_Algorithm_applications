from ising_model import IsingModel
import instance_genrerantion
import numpy as np
import datetime

# This file is for doing the actual simulation and saving the results to be latter annalysed

# compute for a sigle instance
def compute_single_instance(instance_size, step, n_itterations, n_cond_init):
    # generate the instance
    J, H = instance_genrerantion.generate_instance(instance_size)

    # run the algorithm
    ising_model = IsingModel(step, n_itterations, n_cond_init, J, H)
    states, energies = ising_model.simulate()

    # Save all the data (parameters, instance, results) inside of a file
    # Contains the parameters, the instance matrix and the states and enrgies matrices
    now = datetime.datetime.now()
    datestr = now.strftime("%d%m%Y-%H%M")
    filename = f"{instance_size}_{step}_{n_itterations}_{n_cond_init}_{datestr}.npz"
    np.savez(f'computing_results/{filename}', J=J, H=H, states=states, energies=energies)

    return states, energies

if __name__ == '__main__':
    compute_single_instance(instance_size=20, step=0.001, n_itterations=10000, n_cond_init=20)