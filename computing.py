from ising_model import IsingModel
import instance_genrerantion
import numpy as np
import datetime
import concurrent.futures

# This file is for doing the actual simulation and saving the results to be latter annalysed

# compute for a sigle instance
def compute_single_instance(instance_size, step, n_itterations, n_cond_init, J, H, temperature=None, a=None, savetofile=True, stopping_criterion=0, save_history=True, parallel=False, n_threads=10, n_sims_per_thread=10):
    
    #------------------
    # Run the simulation on one thread
    #------------------
    if not parallel:
        ising_model = IsingModel(step, n_itterations, n_cond_init, J, H, temperature, a, stopping_criterion=stopping_criterion, save_history=save_history)
        states, energies, last_energies, biffurcation_rate = ising_model.simulate()
    
    #------------------
    # Run multiples simulations in (true) parallel
    #------------------
    else:
        # Create the ising model with the right number of simulations its gonna run, that we are going to run multiple times in parallel
        ising_model = IsingModel(step, n_itterations, n_sims_per_thread, J, H, temperature, a, stopping_criterion=stopping_criterion, save_history=save_history)

        # Run the simulations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(ising_model.simulate) for _ in range(n_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Unpack the results
        loc_states_arrays, loc_energies_arrays = [], []

        for i in range(n_threads):
            loc_states, loc_energies, _, _, _ = results[i]
            loc_states_arrays.append(loc_states)
            loc_energies_arrays.append(loc_energies)
    
        agg_energies = np.concatenate(loc_energies_arrays, axis=0)
        agg_states = np.concatenate(loc_states_arrays, axis=0)

    #------------------------------------------------
    # Save all the data (parameters, instance, results) inside of a file
    # Contains the parameters, the instance matrix and the states and enrgies matrices
    #------------------------------------------------
    path = ''
    if savetofile:
        now = datetime.datetime.now()
        datestr = now.strftime("%d%m%Y-%H%M")
        filename = f"{instance_size}_{step}_{n_itterations}_{n_cond_init}_{datestr}.npz"
        path = f'computing_results/{filename}'
        np.savez(path, J=J, H=H, states=states, energies=energies)

    return states, energies, last_energies, biffurcation_rate, path