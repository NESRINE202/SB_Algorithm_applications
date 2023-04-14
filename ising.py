import numpy as np
import matplotlib.pyplot as plt
import time

from matrice import n, pas, iteration, n_cond_init, generate_instance

class IsingModel:
    def __init__(self, n_part, pas, iteration,M,H,n_cond_init):
        # n nombre de particule
        # M matrice des forces d'interaction
        # n_cond_init, nombres de conditions initiales différentes
        self.n_part = n_part
        self.pas = pas
        self.iteration = iteration
        self.M =M
        self.H=H
        self.n_cond_init=n_cond_init

    
    def a(self, t): #a(t) dans le document thermal mais je n'ai aps encore compris l'utilite
        return 0 #fonction par hazard

    def temperature(self, t): #fonction pour donner la fluctuation du a la variation de temperature
        return 0.01
    
    def simplectic_update_forall_simulations(self, positions, speeds, t):
        # states of shape (n_cond_init, n_particles, 2)

        # updating the speeds
        forces = -np.dot(self.M, positions.T).T+self.H*positions
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
            states[:, :, t, 0], states[:, :, t, 1] = self.simplectic_update_forall_simulations(prev_positions, prev_speeds, t) # current_positions, current_speeds
            current_positions = states[:, :, t, 0]

            # calcul des énergies
            signed_positions = np.where(current_positions>0, 1, -1)
            current_energies = np.sum(signed_positions @ self.M * signed_positions, axis=1) + self.H @ signed_positions.T # 1d array containing the energies for all the initial conditions
            energies[:, t] = current_energies
        
        return states, energies



if __name__ == "__main__":
    # little example to showcase what it can do

    from performence_analysis import extract_solution

    M, H = generate_instance()
    ising_model = IsingModel(n, pas, iteration, M, H, n_cond_init)
    print("Simulating")
    start_time =time.time()
    states, energies = ising_model.simulate()
    stop_time = time.time()
    print("Done.")
    print(f"Elapse: {stop_time-start_time} seconds")

    solution_energy, spin_configuration, simulaiton_solution_index = extract_solution(states, energies)
    print(f"Minimum energy reached across all the simulations: {solution_energy}.")
    print(f"The first simulation to reach that energy was simulation number {simulaiton_solution_index+1}.")
    print(f"The corresponding spin configuration is: {spin_configuration}")

    abcisses = np.arange(iteration)
    for i in range(n_cond_init):
        plt.plot(abcisses, energies[i])

    plt.title("niveau d'$é$n$é$rgie")    
    plt.xlabel("temps")
    plt.ylabel("E")
    plt.show()