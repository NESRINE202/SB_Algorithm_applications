from reduction_partitio.matrice import n, pas, iteration, nbrsimulation
import numpy as np
from reduction_partitio.ising import M,H




configurations = np.array([[-1] * n])
for i in range(2**n-1):
    binary = np.array(list(bin(i+1)[2:].zfill(n)), dtype=int)
    configurations = np.vstack((configurations, 2*binary-1))

energies = np.dot(configurations, np.dot(M, configurations.T)).diagonal() + np.dot(configurations, H)

best_t1 = configurations[np.argmin(energies)]
min_value = energies.min()

print(best_t1)
print(min_value)
print("M=",M)
print("H=",H)


     