import numpy as np
import matplotlib.pyplot as plt
from matrice import n, pas, iteration, n_cond_init, generate_instance
from ising import IsingModel


# Returns the solution energy, the corresponding spin configuration and the index of the simulation that reached that energy
def extract_solution(states, energies):
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

# Solves n_solves problems and plots a histogram of the number of solutions needed to reach the miminum reach across all simulations
def plot_indexes(n_solves):
    indexes = []

    for i in range():
        M, H = generate_instance()
        ising_model = IsingModel(n, pas, iteration, M, H, n_cond_init)
        states, energies = ising_model.simulate()

        _, _, simulaiton_solution_index = extract_solution(states, energies)
        indexes.append(simulaiton_solution_index+1)

        print(f'Simulaiton {i+1} done')

    print(indexes)

    plt.hist(indexes)
    plt.xlabel("Indice of the first simulation to reach the minimal energy state")
    plt.show()


# Plots the distribution of the energies found on each simulaiton for a same problem
def plot_accoracy():
    # Generate the instance and solve it
    M, H = generate_instance()
    ising_model = IsingModel(n, pas, iteration, M, H, n_cond_init)
    states, energies = ising_model.simulate()


# Histogram of the minimum energies reached by each simulation
def plot_energies_hist(n_cond_init):
    M, H = generate_instance()
    ising_model = IsingModel(n, pas, iteration, M, H, n_cond_init)
    _, energies = ising_model.simulate()

    minimums = energies.min(axis=1)
    plt.hist(minimums)
    plt.show()


if __name__=='__main__':
    plot_energies_hist(1000)