import numpy as np
import matplotlib.pyplot as plt
import time

# This file creats the IsingModel class

class IsingModel:
    def __init__(self, step, iteration, n_cond_init, J, H, custom_temperature=None, custom_a=None):
        # n nombre de particule
        # J matrice des forces d'interaction
        # n_cond_init, nombres de conditions initiales différentes
        self.n_part = len(J)
        self.iteration = iteration
        self.J =J
        self.H=H
        self.temperature_func = custom_temperature if custom_temperature is not None else self.default_temperature
        self.a_func = custom_a if custom_a is not None else self.default_a
        self.step = step if callable(step) else (lambda self, t: step)
        self.n_cond_init=n_cond_init

    
    # Default a and temperature functions
    def default_a(self, t):
        return 0 

    def default_temperature(self, t):
        return 0

    # a and temparature functions callers
    def a(self, t):
        return self.a_func(self, t)
    
    def temperature(self, t):
        return self.temperature_func(self, t)
    
    def simplectic_update_forall_simulations(self, positions, speeds, t):
        # states of shape (n_cond_init, n_particles, 2)

        # updating the speeds
        forces = -np.dot(self.J, positions.T).T-self.H
        speeds = speeds * (1-self.a(t) + self.step(self, t) * self.temperature(t))
        speeds = speeds + self.step(self, t) * forces

        # updating the positions
        positions = positions + self.step(self, t) * speeds
        
        return positions, speeds

    def simulate(self):
        # création des tenseurs
        # cf. détails en photo jointe pour la compréhention de la forme
        states = np.zeros(shape=(self.n_cond_init, self.n_part, self.iteration, 2)) 
        energies = np.zeros(shape=(self.n_cond_init, self.iteration))

        # attribution des conditions initiales
        states[:, :, 0, 0] = np.random.randint(low=0, high=2, size=(self.n_cond_init, self.n_part)) * 2 - 1 # spins initiaux randoms dans {-1, +1}

        # itération
        for t in range(1, self.iteration):
            # positions et vitesses pour toutes les conditions initiales a l'instant t
            prev_positions, prev_speeds = states[:, :, t-1, 0], states[:, :, t-1, 1]
            signed_prev_positions = np.where(prev_positions>1, 1, prev_positions)
            signed_prev_positions = np.where(prev_positions<-1, -1, signed_prev_positions)
            signed_prev_speed = np.where(prev_speeds>1, 1, prev_speeds)
            signed_prev_speed = np.where(prev_speeds<-1, -1, signed_prev_speed)
            states[:, :, t, 0], states[:, :, t, 1] = self.simplectic_update_forall_simulations(signed_prev_positions, prev_speeds, t) # current_positions, current_speeds
            current_positions = states[:, :, t, 0]

            # calcul des énergies  ####### calcule de l'energie n'a pas de sens avec des x variable je pense
            new_signed_pos=np.where(prev_positions>0, 1, -1)
            
            current_energies = np.sum(new_signed_pos @ self.J * new_signed_pos, axis=1) + self.H @ new_signed_pos.T # 1d array containing the energies for all the initial conditions
            energies[:, t] = current_energies
        
        return states, energies


if __name__ == "__main__":
    import instance_genrerantion
    from results_annalysis import plot_energies_evolution
    #plot_energies_evolution() 