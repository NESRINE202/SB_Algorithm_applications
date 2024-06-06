import numpy as np
from simulation_manager import SimulationManager
import results_annalysis
import matplotlib.pyplot as plt
# Useful functions to the reduction 

def rotation_lignes(Matrix,n): 
    
    for _ in range(n):
        Ligne_0 = Matrix[0,:].copy()
        for i in range(len(Matrix)- 1): 
            Matrix[i,:] = Matrix[i+1,:]

        Matrix[-1,:] = Ligne_0

    return Matrix


def C_rotation(Size): 
    
    c_n = np.zeros((Size,Size))
    for i in range(Size-1): 
        c_n+=rotation_lignes(np.eye(Size),i+1)

    c = np.kron(np.eye(Size), c_n)

    return c

def permutation_matrix(Size):
    M =np.random.permutation(Size*Size).reshape((Size,Size))
    X = np.array([M[i,j] for i in range(Size) for j in range(Size) ])
    f_X = np.array([M[i,j] for j in range(Size) for i in range(Size) ])
    n = len(X)
    P = np.zeros((n, n), dtype=int)  
    for i in range(n):
        index = np.where(f_X==X[i])[0][0]
        P[index][i] = 1

    return P

class arbitrage(): 
    def __init__(self,exchange_matrix) -> None:
        self.exchange_matrix = exchange_matrix
        
    def arbitrage_reduction(self,Taux,M1 =1,compact=False):
        """
        Input: Exchange rate matrice C
        output: Symmetrical part of J and H  if compact is false 
                else Compact with H integrated and H 
        """
        log_vectorized = np.vectorize(np.log)
        log_Taux = log_vectorized(Taux)
        
        Size = len(log_Taux)
        # Defining the reduction matrices 
        C = C_rotation(Size)
        P = permutation_matrix(Size)

        # Creating H_1

        H_1= -np.array([log_Taux[i,j] for i in range(len(log_Taux)) for j in range(len(log_Taux[0]))])
        I = np.ones(len(H_1))
        
        # Creating J_1
        J_1 = 2*C + 2* P.T@C@P+ 2*np.eye(len(C)) - P - C@P

        H = H_1/2 + M1*(J_1.T+J_1)@I /4
        J = -M1*J_1/2

        # Extract the symmetrical part of J 
        J_sym =(J+J.T)/2 
        np.fill_diagonal(J_sym,0)

        if not compact: 
        # Needs to be verified(Is it necessary?)
            return  J_sym,H
        else: 
            # In here we will follow the methods used in paper 2023
            J_compact = np.zeros((len(J)+1,len(J)+1))
            J_compact[:len(J),:len(J)] = J_sym
            J_compact[:len(J),len(J)] = H 
            J_compact[len(J),:len(J)] = H
            H_compact = np.zeros(len(H)+1)
            # H_compact[:len(H)] = H

            return J_compact,H_compact
        
    def find_optimal_cycle(self,visualize = False): 
        J,H  = self.arbitrage_reduction(self.exchange_matrix)
        eigs = np.linalg.eigvals(J)
        norm_eigs = np.abs(eigs)
        ksi = 1/(norm_eigs.max())
        p_first = 1-ksi*np.real(eigs.max())
        p_last = 1-ksi*np.real(eigs.min())

        step_size = 0.01
        num_iterations= 2000
        num_simulations= 50
        lag = 0 # Remember what the lag is for 

        def pumping_rate(self, t):
            # return t/100
            if t<lag:
                return 0
            else:
                return 1 * p_last * (t-lag)/(num_iterations-lag)
            
        manager = SimulationManager(step_size=0.1, num_iterations=num_iterations, num_simulations=2, J=J, H=H, pumping_rate=pumping_rate, stopping_criterion=0.1, save_states_history=True, save_energies_history=True, n_threads=2, savetofile=False)
        states, energies, last_states, last_TAC_states, last_sign_States, last_energies, last_TAC_energies, last_sign_energies, final_times, sign_times, TAC_times = manager.run_simulation()
        solution = np.sign(last_states[0, :, 0])
        if visualize : 
            results_annalysis.complete_plot(states, energies, 0)
        return solution 