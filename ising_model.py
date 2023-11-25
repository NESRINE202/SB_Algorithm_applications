import numpy as np
import matplotlib.pyplot as plt
import time

# This file creats the IsingModel class

class IsingModel:
    def __init__(self, pas, iteration, n_cond_init, J, H):
        # n nombre de particule
        # J matrice des forces d'interaction
        # n_cond_init, nombres de conditions initiales différentes
        self.n_part = len(J)
        self.pas = pas
        self.iteration = iteration
        self.J =J
        self.H=H
        self.n_cond_init=n_cond_init

    
    def a(self, t): #a(t) dans le document thermal mais je n'ai aps encore compris l'utilite
        return 0 #fonction par hazard

    def temperature(self, t): #fonction pour donner la fluctuation du a la variation de temperature
        return 0.01
    
    def simplectic_update_forall_simulations(self, positions, speeds, t):
        # states of shape (n_cond_init, n_particles, 2)

        # updating the speeds
        forces = -np.dot(self.J, positions.T).T-self.H #*positions y avais aussi un plus avec le H j'ai mis -
        speeds = speeds * (1-self.a(t) + self.pas * self.temperature(t))
        speeds = speeds + self.pas * forces

        # updating the positions
        positions = positions + self.pas * speeds
        
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