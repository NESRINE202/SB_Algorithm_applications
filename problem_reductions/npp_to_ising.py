import numpy as np

# Creating the interation matrix from a given NPP instance
def npp_to_ising(weights):
    J = np.outer(weights, weights)
    np.fill_diagonal(J, 0)
    H = np.zeros(len(weights))
    return J, H

# Returning the two partitions from the given NPP instance and the spin configuration yielded from the Ising reduction
def ising_to_npp(spins, partition):
    indixes_set_1 = np.where(spins==1)
    indixes_set_2 = np.where(spins==-1)

    partition_1 = partition[indixes_set_1]
    partition_2 = partition[indixes_set_2]

    return partition_1, partition_2

# returns the cost function from a given solution
def configuration_to_cost(spins, partition):
    partition_1, partition_2 = ising_to_npp(spins, partition)
    cost = np.abs(partition_1.sum() - partition_2.sum())

    return cost