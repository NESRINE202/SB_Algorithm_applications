import numpy as np
import computing, results_annalysis
import instance_genrerantion
import verifbrute
import matplotlib.pyplot as plt

sb_energies = []
exact_energies = []

for n in range(3, 16):
    instance_size = 100
    J, H = instance_genrerantion.generate_instance(size=n)
    _, energies, _ = computing.compute_single_instance(instance_size=instance_size, step=0.001, n_itterations=10000, n_cond_init=50, J=J, H=H, savetofile=False)
    sb_energy = energies.min()
    exact_energy, _ = verifbrute.verif_function(J, H)

    sb_energies.append(sb_energy)
    exact_energies.append(exact_energy)

exact_energies = np.array(exact_energies)
sb_energies = np.array(sb_energies)

relative_errors = np.abs(exact_energies-sb_energies)/exact_energies

np.savez('results.npz', relative_errors=relative_errors, exact_energies=exact_energies, sb_energies=sb_energies)