import numpy as np

# Creating the interation matrix from a given NPP instance
def npp_to_ising(partition):
    J = -np.outer(partition, partition)
    return J

# Returning the two partitions from the given NPP instance and the spin configuration yielded from the Ising reduction
def ising_to_npp(spins, partition):
    indixes_set_1 = np.where(spins==1)
    indixes_set_2 = np.where(spins==-1)

    partition_1 = partition[indixes_set_1]
    partition_2 = partition[indixes_set_2]

    return partition_1, partition_2