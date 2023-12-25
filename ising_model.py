import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from joblib import Parallel, delayed

# This file creats the IsingModel class

class IsingModel:
    def __init__(self, step, iteration, n_cond_init, J, H, custom_temperature=None, custom_a=None, stopping_criterion=0, save_history=True):
        # n nombre de particule
        # J matrice des forces d'interaction
        # n_cond_init, nombres de conditions initiales différentes
        # stopping_criterion, % of particles that need to have bifurcated before stopping the algorithm
        self.n_part = len(J)
        self.iteration = iteration
        self.stopping_criterion = stopping_criterion
        self.J =J
        self.H=H
        self.save_history = save_history
        self.temperature_func = custom_temperature if custom_temperature is not None else self.default_temperature
        self.a_func = custom_a if custom_a is not None else self.default_a
        self.step = step if callable(step) else (lambda self, t: step)
        self.n_cond_init=n_cond_init

    
    # Default a and temperature functions
    def default_a(self, t, _):
        return 0 

    def default_temperature(self, t, _):
        return 0

    # a and temparature functions callers
    def a(self, t):
        return self.a_func(self, t)
    
    def temperature(self, t):
        return self.temperature_func(self, t)
    
    #----------------------------------
    # CASE 1: Solving of the Eurler scheme
    #----------------------------------
    def simplectic_update_forall_simulations(self, positions, speeds, t):
        # states of shape (n_cond_init, n_particles, 2)

        # updating the speeds
        forces = -np.dot(self.J, positions.T).T - self.H
        # speeds = speeds * (1-self.a(t) + self.step(self, t) * self.temperature(t))
        speeds = speeds + 0.09490117387887338 * self.step(self, t) * forces + (-1 + self.temperature(t) - np.square(positions))*positions

        # updating the positions
        positions = positions + self.step(self, t) * speeds

        # Implementing the walls at -1 and +1
        positions = np.clip(positions, -1, 1)
        speeds = np.where((positions == 1) | (positions == -1), 0, speeds)
        
        return positions, speeds
    
    def simplectic_update_forall_simulations_parallel(self, positions, speeds, t):
        # states of shape (n_cond_init, n_particles, 2)

        # updating the speeds
        forces = -parallel_multiply(self.J, positions) - self.H
        speeds = speeds * (1-self.a(t) + self.step(self, t) * self.temperature(t))
        speeds = speeds + self.step(self, t) * forces

        # updating the positions
        positions = positions + self.step(self, t) * speeds

        # Implementing the walls at -1 and +1
        positions = np.clip(positions, -1, 1)
        speeds = np.where((positions == 1) | (positions == -1), 0, speeds)
        
        return positions, speeds
    

    
    def simulate(self):
        energies = np.zeros(shape=(self.n_cond_init, self.iteration))
        biffurcation_rate = np.ones(shape=(self.iteration))

        #----------------------------------
        # CASE 1: Compute and save the state at each iteration (slower, uses more memory)
        #----------------------------------
        if self.save_history:
            #----------------------------------
            # Define the arrays that will hold the state
            #----------------------------------
            # 3D here because we save the state at each iteration
            states = np.zeros(shape=(self.n_cond_init, self.n_part, self.iteration, 2))
            # states[:, :, 0, 0] = np.random.randint(0, 2, size=(self.n_cond_init, self.n_part)) * 2 - 1
            # states[:, :, 0, 0] = np.random.uniform(-1, 1, size=(self.n_cond_init, self.n_part))
            # states[:,:,0,0]= np.cos(np.random.uniform(0, 2*np.pi, size=(self.n_cond_init,self.n_part)))
            states[:, :, 0, 0] = np.random.normal(0, 0.0001, size=(self.n_cond_init, self.n_part))
            # states[:, :, 0, 0] = np.clip(states[:, :, 0, 0], -1, 1)
            # states[:, :, 0, 0] = np.zeros(shape=(self.n_cond_init, self.n_part))

            # Iterrate over time
            for t in range(1, self.iteration):
                #----------------------------------
                # Update de positions and speeds using Euler's scheme
                #----------------------------------
                prev_positions, prev_speeds = states[:, :, t-1, 0], states[:, :, t-1, 1]
                states[:, :, t, 0], states[:, :, t, 1] = self.simplectic_update_forall_simulations(prev_positions, prev_speeds, t)
                current_positions = states[:, :, t, 0]
                
                #----------------------------------
                # Current state energy computation
                #----------------------------------
                current_signed_pos = np.where(current_positions > 0, 1, -1)
                current_energies = np.sum(current_signed_pos @ self.J * current_signed_pos, axis=1) + self.H @ current_signed_pos.T
                energies[:, t] = current_energies

                #--------------------------
                # Global stoping criterion
                #--------------------------
                # Count the number of particles (over all simulations), that have not yet bifurcated
                positions, speeds = (states[:, :, t, 0], states[:, :, t, 1])
                mask = ((positions == -1) | (positions == 1)) & (speeds == 0)
                number_of_parts_affected = len(positions[mask])
                biffurcation_rate[t] = number_of_parts_affected / (self.n_part * self.n_cond_init)

                # Stop the algorithm if the stopping criterion is reached
                if (biffurcation_rate[t] >= 1 - self.stopping_criterion) and (t>=3):
                    return states[:, :, :t+1, :], energies[:, :t+1], 0, biffurcation_rate[:t+1]

            return states, energies, 0, biffurcation_rate

        #----------------------------------
        # CASE 2: Compute without saving the states (much faster for high number of simulations, uses less memory)
        #----------------------------------
        else:
            #----------------------------------
            # Define the arrays that will hold the state
            #----------------------------------
            # 2D here because we only save the last states 
            current_state = np.zeros(shape=(self.n_cond_init, self.n_part, 2))
            # current_state[:, :, 0] = np.random.randint(0, 2, size=(self.n_cond_init, self.n_part)) * 2 - 1
            # current_state[:, :, 0] = np.random.uniform(-1, 1, size=(self.n_cond_init, self.n_part)) * 2 - 1
            current_state[:, :, 0] = np.zeros(shape=(self.n_cond_init, self.n_part))
            min_energies = np.zeros(shape=(self.iteration))

            # Iterrate over time
            for t in range(1, self.iteration):
                #----------------------------------
                # Update de positions and speeds using Euler's scheme
                #----------------------------------
                prev_positions, prev_speeds = current_state[:, :, 0], current_state[:, :, 1]
                current_state[:, :, 0], current_state[:, :, 1] = self.simplectic_update_forall_simulations(prev_positions, prev_speeds, t)
                current_positions = current_state[:, :, 0]

                #----------------------------------
                # Current state energy computation
                #----------------------------------
                current_signed_pos = np.where(current_positions > 0, 1, -1)
                current_energies = np.sum(current_signed_pos @ self.J * current_signed_pos, axis=1) + self.H @ current_signed_pos.T
                current_min_energy = current_energies.min()
                min_energies[t] = current_min_energy
                # energies[:, t] = current_energies

                #--------------------------
                # Global stoping criterion
                #--------------------------
                # Count the number of particles (over all simulations), that have not yet bifurcated
                positions, speeds = (current_state[:, :, 0], current_state[:, :, 1])
                mask = (positions != -1) & (positions != 1) & (speeds != 0)
                number_of_parts_affected = len(positions[mask])
                biffurcation_rate[t] = number_of_parts_affected / (self.n_part * self.n_cond_init)

                # Stop the algorithm if the stopping criterion is reached
                if biffurcation_rate[t] <= self.stopping_criterion:
                    best_sim_index = np.argmin(current_energies)
                    
                    return current_state[best_sim_index], min_energies[:t+1], current_energies, biffurcation_rate[:t+1], 
            
            return current_state, min_energies, biffurcation_rate
        

def multiply_row(J, row):
        return np.dot(J, row)

# def parallel_multiply(J, positions, num_processes=None):
#     with Pool(processes=num_processes) as pool:
#         results = pool.starmap(multiply_row, [(J, row) for row in positions])
#     return np.array(results)

def parallel_multiply(J, positions, num_jobs=-1):
    results = Parallel(n_jobs=num_jobs)(delayed(multiply_row)(J, row) for row in positions)
    return np.array(results)

def regular_multiply(J, positions):
    return np.dot(J, positions.T).T