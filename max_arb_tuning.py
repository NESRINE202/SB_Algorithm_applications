import numpy as np
import simulation_manager, results_annalysis
import matplotlib.pyplot as plt
import pandas as pd

def penalties_and_proffit(S, J):
    # P1 calculation
    P1 = np.sum(np.sum(S * S.T, axis=1) - np.diag(S * S.T))

    # P2 calculation
    P2 = np.sum(np.sum(S * S.T, axis=0) - np.diag(S * S.T))

    # P3 calculation
    sum_row = np.sum(S, axis=1)
    sum_col = np.sum(S, axis=0)
    P3 = np.sum((sum_row - sum_col) ** 2)

    # P4 calculation
    P4 = np.sum(S * S.T)

    coeffs = J
    coeffs[coeffs>=0] = 0
    proffit = np.exp(-(coeffs * S).sum()) - 1

    return [P1, P2, P3, P4, proffit, P1+P2+P3+P4-proffit]

def solve_max_arb(J, H, lambdas, num_iterations, step, mask, plot):

    def pumpuing_rate(self, t):
            return t/num_iterations
    
    #------- running simulation -----------
    manager = simulation_manager.SimulationManager(step_size=step, num_iterations=num_iterations, num_simulations=1, J=J, H=H, pumping_rate=pumpuing_rate, stopping_criterion=0, save_states_history=True, save_energies_history=False, n_threads=1, savetofile=False, lambdas=lambdas)
    states, _, last_states, _ = manager.run_simulation()
    
    solutions = last_states[:, :, 0]

    #------- shaping solutions -----------
    shaped_solutions = []
    for i in range(solutions.shape[0]):
        shaped_solution = solutions[i].reshape((5,5))
        shaped_solution = (shaped_solution+1)/2
        # shaped_solution = shaped_solution * mask
        shaped_solutions.append(shaped_solution)

    shaped_solutions = np.array(shaped_solutions)

    #------- penalties and proffit -----------
    results = []

    for i in range(solutions.shape[0]):
        S = shaped_solutions[i]
        results.append(penalties_and_proffit(S, J))

    results = np.array(results)

    #------- % of good solutions -----------
    good_sols_counter = 0
    for result in results:
        if list(result[:4]) == [0, 0, 0 ,0]:
            good_sols_counter += 1

    percentage = good_sols_counter/results.shape[0]*100
    results[0]

    #------- plotting results -----------
    # First Plot
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot

        positions = states[:, :, :, 0]
        _, n_particle, n_iterration = positions.shape
        abcisses = np.arange(n_iterration)

        for i in range(n_particle):
            pos = positions[0, i, :]
            plt.plot(abcisses, pos)
            plt.xlabel("Iteration number")
            plt.ylabel("Particle position")
            plt.title("Particle Position Evolution")

        # Second Plot
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot

        pen_proff = []
        for t in range(num_iterations):
            S = states[0, :, t, 0].reshape((5, 5))
            S = (S + 1) / 2
            pen_proff.append(penalties_and_proffit(S, J))

        pen_proff = np.array(pen_proff).T

        plt.plot(abcisses, pen_proff[0])
        plt.plot(abcisses, pen_proff[1])
        plt.plot(abcisses, pen_proff[2])
        plt.plot(abcisses, pen_proff[3])
        plt.plot(abcisses, pen_proff[4])
        plt.legend(['P1', 'P2', 'P3', 'P4', 'Profit'])
        plt.xlabel("Iteration number")
        plt.title("Penalties and Profit Evolution")

        plt.tight_layout()  # Adjust the layout
        plt.show()


    return results[0], states[0, :, t, 0].reshape((5, 5))