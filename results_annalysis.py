import numpy as np
import matplotlib.pyplot as plt
from instance_genrerantion import n, pas, iteration, n_cond_init, generate_instance
from ising_model import IsingModel
import time

# This file is for reading results saved by the computing.py file and to make various visualization of it

# HOW TO IMPLEMET NEW PLOTS: Build a function of (states, energies) that ends in some sort of plt.show(), then run this function from the if __name__ == "__main__"

def open_results(path):
    # unpack the parameters used for the simulation
    instance_size, step, n_itterations, n_cond_init = path[18:].split('_')[:-1]

    # load and unpack the datas
    data = np.load(path)
    J, H = data['J'], data['H']
    states, energies = data['states'], data['eneries']

    return states, energies

# energies evolution
def plot_eneries_evolution(states, energies):
    n_cond_init = len(energies)
    n_iterration = len(energies[0])
    abcisses = np.arange(n_iterration)
    for i in range():
        plt.plot(abcisses, energies[i])

    plt.show()

# Histogram of the minimum energies reached by each simulation
def plot_energies_hist(states, energies):

    minimums = energies.min(axis=1)
    plt.hist(minimums)
    plt.show()

# Returns the solution energy, the corresponding spin configuration and the index of the simulation that reached that energy
def extract_full_solution(states, energies):
    mins_indexes = np.argmin(energies, axis=1)
    solution_energy = np.min(energies)
    simulaiton_solution_index = 0

    for i in range(len(mins_indexes)):
        iteration_index = mins_indexes[i]
        if energies[i, iteration_index] == solution_energy:
            simulaiton_solution_index = i

    min_coord = simulaiton_solution_index, mins_indexes[simulaiton_solution_index]

    spin_configuration = np.where(states>0, 1, -1)
    spin_configuration = spin_configuration[min_coord[0], :, min_coord[1], 0]
    spin_configuration = spin_configuration.flatten()

    return solution_energy, spin_configuration, simulaiton_solution_index


if __name__=='__main__':
    plot_energies_hist(1000)