import numpy as np
import simulation_manager, results_annalysis
import matplotlib.pyplot as plt
import time
import pandas as pd

class OptimalPortfolio(): 
    def __init__(self,N,K,sigma,mu,alpha):
        """
        Defining the parameters
        N: number of assets to choose from 
        K: the budget expressed with units 
        sigma: covariance matrix between assets 
        mu: return vector of assets 
        alpha: risk aversion parameter 
        """
        self.N = N
        self.K =K
        self.sigma = sigma
        self.mu = mu 
        self.alpha = alpha 

    
    def P_aug(self,N,bit): 
        P = np.zeros((N,N*bit))
        b = np.array([2**i for i in range(bit)])
        for i in range(N):
            debut = i*bit
            fin = debut + bit 
            P[i,debut:fin] = b 

        return P 
   

    def reduction_markovitz(self):
        """
        From the covariance matrix and the excpected return vector returns the J and H 
        of the Ising mapping 

        ARGS: 
        sigma: covariance matrix 
        mu: excpected return vector
        alpha : aversion constant 
        K : The budget 
        penalty: If we add the constraint in the energy 
        Output: 
        J : 
        H: 
        """
        k = self.K/self.N
        Bit_max = int(np.log2(k))
        bit = Bit_max
        N = len(self.mu) 
        U = np.ones(bit*N) 
        P = self.P_aug(N,bit)
        sigma_augmented = P.T @ self.sigma @ P
        mu_augmented = P.T @ self.mu 

        J = - (self.alpha/2)* sigma_augmented
        H = (self.alpha/2) * sigma_augmented @ U - mu_augmented 
        np.fill_diagonal(J,0)

        return J,H
        
    
    def reverse(self,solution): 
        solution += 1
        solution /=2
        bit = int(np.log2(self.K/self.N))
        P = self.P_aug(self.N,bit)
        return P @ solution
        
    def find_optimal_portfolio(self,visualize = False): 
        # defining SB parameters 

        J,H = self.reduction_markovitz()
        eigs = np.linalg.eigvals(J)

        ksi = 1/(eigs.max())
        p_first = 1-ksi*np.real(eigs.max())
        p_last = 1-ksi*np.real(eigs.min())
        num_iterations = 1000
        lag = 0

        def pumpuing_rate(self, t):
            # return t/100
            if t<lag:
                return 0
            else:
                return 1 * p_last * (t-lag)/(num_iterations-lag)
            
        manager = simulation_manager.SimulationManager(step_size=0.5, num_iterations=num_iterations, num_simulations=2, J=J, H=H, pumping_rate=pumpuing_rate, stopping_criterion=0.1, save_states_history=True, save_energies_history=True, n_threads=2, savetofile=False)

        states, energies, last_states, last_TAC_states, last_sign_States, last_energies, last_TAC_energies, last_sign_energies, final_times, sign_times, TAC_times = manager.run_simulation()
        if visualize : 
            results_annalysis.complete_plot(states, energies, 0)

        solution= np.sign(last_states[0, :, 0])

        return self.reverse(solution)

