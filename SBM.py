import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from joblib import Parallel, delayed
import random

# This file creats the IsingModel class

class SBM:
    def __init__(self, J, H, step_size, num_iterations, num_simulations, stopping_criterion=0, save_states_history=True, save_energies_history=True, custom_pumping_rate=None):
        """
        Args:
            ------Numerical resultion parameters-------
            step_size (float): Step size for the Euler scheme.
            num_iterations (int): Number of iterations to run the simulation for at maximum.
            num_simulations (int): Number of simulations to run with this SBM.
            stopping_criterion (float): % of the particles that have to reach the stopping criterion for the simulation to stop.
            save_history (bool): Whether to save the history of the simulation or not.

            ------------Physical parameters--------------
            J (numpy.ndarray): Array of shape (num_particles, num_particles) representing the interactions between the particles.
            H (numpy.ndarray): Array of shape (num_particles) representing the external field applied on the particles.
            custom_pumping_rate (function): Two photo pumping rate.
            
        """
        self.num_particles = J.shape[0]**2
        self.num_iterations = num_iterations
        self.stopping_criterion = stopping_criterion
        self.J =J
        self.H=H
        self.save_states_history = save_states_history
        self.save_energies_history = save_energies_history
        self.pumping_rate_func = custom_pumping_rate if custom_pumping_rate is not None else self.default_pumping_rate
        self.step_size = step_size if callable(step_size) else (lambda self, t: step_size)
        self.num_simulations=num_simulations
        self.initialize_model()


    def default_pumping_rate(self, t, _):
        """
        Default pumping rate function in case no argument is given.
        """
        return 0
    
    def pumping_rate(self, t):
        """
        Caller for the pumping rate function. Weather it is the detault one or the one specified in the class builder arguments.
        """
        return self.pumping_rate_func(self, t)
    
    def initialize_model(self):
        """
        Initializes the model for the simulation.

        Args:
            None

        Returns:
            None

        Additional notes: It initializes following
            - The full states array (if asked to)
            - The model numerical parameters such as ksi and other
            - The initial conditions for the simulations
        """

        #-------- Arrays to store states, current state and energies -------
        if self.save_states_history:
            self.states = np.zeros(shape=(self.num_simulations, self.num_particles, self.num_iterations, 2))
        if self.save_energies_history:
            self.energies = np.zeros(shape=(self.num_simulations, self.num_iterations))

        self.current_state = np.zeros(shape=(self.num_simulations, self.num_particles, 2))
        self.energies = np.zeros(shape=(self.num_simulations, self.num_iterations))

        #-------- Model Parameters --------
        self.ksi = 0.5/np.sqrt( np.sum(np.square(self.J)) / (self.num_particles-1) )

        #------ Initial conditions -------
        self.current_state[:, :, 0] = np.random.normal(0, 0.001, size=(self.num_simulations, self.num_particles))


    def simplectic_update(self, positions, speeds, t):
        """
        Update of the speeds/positions based on the current state and the described system dynamics. Solves the Euler scheme.

        Args:
            positions (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the positions of the particles for each simulation.
            speeds (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the speeds of the particles for each simulation.

        Returns:
            positions (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the updated positions of the particles for each simulation.
            speeds (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the updated speeds of the particles for each simulation.
        """

        #------- For manual max-arb --------
        original_shape = positions.shape
        sqrt_num_particles = int(np.sqrt(self.num_particles))
        M = positions.reshape(self.num_simulations, sqrt_num_particles, sqrt_num_particles)
        MT = np.transpose(M, axes=(0, 2, 1))
        # Objective function
        # forces = self.J
        # Contraints
        # P3
        forces = -2*np.sum(M - MT)
        # P2
        forces -= self.num_particles * M
        # P1
        forces -= self.num_particles * M
        # P4
        forces -= MT

        positions = M.reshape(original_shape[0], -1)
        forces = forces.reshape(original_shape[0], -1)

        #-------- Gradient of the potential energy --------
        # forces = -np.dot(self.J, positions.T).T * self.ksi
        
        #-------- For CIM amplitude dynamics --------
        # forces += (-1 + self.pumping_rate(t) - np.square(positions)) * positions

        #-------- For b(alistic)SB simulation  dynamics --------
        # forces += (-1 + self.pumping_rate(t)) * positions

        #-------- For d(iscrete)SB simulatio  dynamics --------
        # forces = -np.dot(self.J, np.sign(positions).T).T * self.ksi + (-1 + self.pumping_rate(t)) * positions # for d(iscrete)SB simulation
        
        # Update speeds and positions
        speeds = speeds + self.step_size(self, t) * forces
        positions = positions + self.step_size(self, t) * speeds

        # Implement the walls at +1 and -1
        positions = np.clip(positions, -1, 1)
        speeds = np.where((positions == 1) | (positions == -1), 0, speeds)
        
        return positions, speeds
    
    def compute_energies(self, positions):
        """
        Computes the different energies of the different simulations based on the positions of the particles.

        Args:
            positions (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the positions of the particles for each simulation.

        Returns:
            current_energies (numpy.adarray): Array of shape (n_simulations) representing the energies of the different simulations.

        Additional notes: This computes the energies of the binarized states.
        """

        # signed_positions = np.where(positions > 0, 1, -1)
        # current_energies = np.sum(signed_positions @ self.J * signed_positions, axis=1) + self.H @ signed_positions.T

        return 0

    def TAC(self, positions, speeds):
        return positions, speeds


    def simulate(self):
        """
        Runs the simulation for the given parameters.
        
        Returns:
            states (numpy.ndarray): Array of shape (n_simulations, num_particles, n_iterations, 2) representing the states of the different simulations at every iteration time.
            energies (numpy.ndarray): Array of shape (n_simulations, n_iterations) representing the energies of the different simulations at every iteration time.
            current_state (numpy.ndarray): Array of shape (n_simulations, num_particles, 2) representing the states of the different simulations at the last iteration time.
            current_energies (numpy.ndarray): Array of shape (n_simulations) representing the energies of the different simulations at the last iteration time.
        """

        for t in range(self.num_iterations):
            # fetch the previous positions and speeds
            positions, speeds = self.current_state[:, :, 0], self.current_state[:, :, 1]

            # Update the positions and speeds
            self.current_state[:, :, 0], self.current_state[:, :, 1] = self.simplectic_update(positions, speeds, t)

            # Compute the new energies
            current_energies = self.compute_energies(positions)

            # Save what needs to be saved
            if self.save_states_history:
                self.states[:, :, t, :] = self.current_state
            if self.save_energies_history:
                self.energies[:, t] = current_energies

        # Return what needs to be returned
        if not self.save_energies_history:
            self.energies = None
        if not self.save_states_history:
            self.states = None
        
        return self.states, self.energies, self.current_state, current_energies