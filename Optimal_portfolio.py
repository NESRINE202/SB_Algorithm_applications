import numpy as np
import simulation_manager, results_annalysis
import matplotlib.pyplot as plt
import time
import pandas as pd

def P_aug(N,bit): 
    P = np.zeros((N,N*bit))
    b = np.array([2**i for i in range(bit)])
    for i in range(N):
        debut = i*bit
        fin = debut + bit 
        P[i,debut:fin] = b 

    return P 
class OptimalPortfolio(): 
    def __init__(self,N,K,sigma,mu,alpha=1):
        """
        Defining the parameters
        N: number of assets to choose from 
        K: the budget expressed with units (K/N should be superroior that 100 )
        sigma: covariance matrix between assets 
        mu: return vector of assets 
        alpha: risk aversion parameter 
        """
        self.N = N
        self.K =K
        self.sigma = sigma
        self.mu = mu 
        self.alpha = alpha 

   

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
        P = P_aug(N,bit)
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
        P = P_aug(self.N,bit)
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
    
    def Evaluation(self,nb_simulations= 10000):

        Optimal= self.find_optimal_portfolio()
        optimal_weights= Optimal/sum(Optimal)
        risque = []
        retour = []

        for _ in range(10000): 
            w = np.random.uniform(0, 75, self.N)
            w /= sum(w)
            wsw = w.T @ self.sigma @ w
            wmu = w.T @ self.mu

            risque.append(wsw)
            retour.append(wmu)

        plt.scatter(risque, retour)
        risque_SB = optimal_weights.T @ self.sigma @ optimal_weights 
        return_SB = optimal_weights.T @ self.mu 
        plt.plot(risque_SB, return_SB, 'ro', label='SB portfolio')  # Plot the SB portfolio point
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.title('Portfolio Risk vs. Return')
        plt.legend()
        plt.show()



class OptimalTrajectory():
    def __init__(self,N,K,T,sigma_t,mu_t,alpha=1,c=0.01):
        """
        Defining the parameters
        N: number of assets to choose from 
        K: the budget expressed with units (K/N should be superroior that 100 )
        T: Time increments
        sigma_t: covariance matrix between assets at T time increments
        mu_t: return vector of assets at T time increments
        alpha: risk aversion parameter ()
        c: fixed transaction cost 
        """
        self.N = N
        self.K =K
        self.T = T
        self.sigma_t = sigma_t
        self.mu_t = mu_t
        self.alpha = alpha 
        self.c= c 

    def reduction(self,sigma,mu,previous_solution = 0,first= False):
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
        k= self.K/self.N
        Bit_max = int(np.log2(k))
        bit = Bit_max
        N = len(mu) # Number of assets considered 
        U = np.ones(bit*self.N) 
        P = P_aug(self.N,bit)
        sigma_augmented = P.T @ sigma @ P
        mu_augmented = P.T @ mu 
        lambda_augmented = self.c*P.T@P
        if not first: 
            J = - (self.alpha/2)* sigma_augmented- lambda_augmented/2
            H = (self.alpha/2) * sigma_augmented @ U - mu_augmented - (1/2)*lambda_augmented @previous_solution
            np.fill_diagonal(J,0)
            return  J,H
        else: 
            J_first = - (self.alpha/2)* sigma_augmented
            H_first = (self.alpha/2) * sigma_augmented @ U - mu_augmented
            np.fill_diagonal(J_first,0)
            return J_first,H_first
        
    
    
    def reverse(self,solution): 
        solution += 1
        solution /=2
        bit = int(np.log2(self.K/self.N))
        P = P_aug(self.N,bit)
        return P @ solution

    
    def find_optimal_portfolio(self,sigma,mu,previous_solution= 0,first =False,visualize = False):
        """
        Find optimal portfolio using Simulated bifurcations 
        Args: 
        sigma : 
        mu : 
        previous_solution 
        first : 
        visualize: 
        """

        
        J,H = self.reduction(sigma,mu,previous_solution,first)

        # Defining the Pumping rate 
        eigs = np.linalg.eigvals(J)
        norm_eigs = np.abs(eigs)
        ksi = 1/(norm_eigs.max())
        p_first = 1-ksi*np.real(eigs.max())
        p_last = 1-ksi*np.real(eigs.min())

        num_iterations= 2000
        lag = 0 

        def pumping_rate(self, t):
            # return t/100
            if t<lag:
                return 0
            else:
                return 1 * p_last * (t-lag)/(num_iterations-lag)
            
        # 
        manager = simulation_manager.SimulationManager(step_size=0.5, num_iterations=num_iterations, num_simulations=2, J=J, H=H, pumping_rate=pumping_rate, stopping_criterion=0.1, save_states_history=True, save_energies_history=True, n_threads=2, savetofile=False)
        states, energies, last_states, _, _, _, _, _, _, _, _ = manager.run_simulation()

        if visualize: 
            results_annalysis.complete_plot(states, energies, 0)

        return np.sign(last_states[0, :, 0])
    
    def find_Trajectory(self):

        solution_0= self.find_optimal_portfolio(self.sigma_t[:,:,0],self.mu_t[:,0],first = True)
        Trajectory_solutions = [solution_0]


        for t in range(1,self.T): 
            solution_t = self.find_optimal_portfolio(self.sigma_t[:,:,t],self.mu_t[:,t],previous_solution=Trajectory_solutions[t-1])
            Trajectory_solutions.append(solution_t)

        Trajectory_weights= [self.reverse(solution) for solution in Trajectory_solutions]
       
        return Trajectory_weights
    

    
    def calculate_trajectory_value(self,Trajectory_weights):
        T = len(Trajectory_weights)
        total_value = np.zeros(T)
        return_component = np.zeros(T)
        risk_component = np.zeros(T)
        
        for t in range(T):
            sigma = self.sigma_t[:,:,t]
            mu = self.mu_t[:,t]
            S = Trajectory_weights[t]
            
            return_component[t] = S.T @ mu
            risk_component[t] = S.T @ sigma @ S 
            if t>1: 
                trading_costs = self.c * np.sum(np.abs(Trajectory_weights[t]- Trajectory_weights[t-1]))
            else: 
                trading_costs = 0
            total_value[t] = return_component[t] - risk_component[t] - trading_costs
        
        return total_value, return_component, risk_component
    
    def Evaluation(self,nb_simulation = 10000):
        # Creating the universe of possible trajectories
        Trajectory_weights = self.find_Trajectory()
        trajectories_universe =[]
        for _ in range(nb_simulation):
            trajectory = np.random.uniform(0,self.K,size=(len(Trajectory_weights),self.N))
            trajectories_universe.append(trajectory)

        # Plotting 
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        for i in range(10000):
            total_value,return_component,risk_component = self.calculate_trajectory_value(trajectories_universe[i])
            axes[0].plot(total_value,'b')
            axes[1].plot(return_component,'b')
            axes[2].plot(risk_component,'b')
        total_value,return_component,risk_component = self.calculate_trajectory_value(Trajectory_weights)

        # Plot total value
        axes[0].plot(total_value, 'r',linewidth = 5, label='Total Value')
        axes[0].set_title(f'Total Value ')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')

        # Plot return component
        axes[1].plot(return_component, 'r',linewidth = 5, label='Return Component')
        axes[1].set_title(f'Return Component ')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Value')

        # Plot risk component
        axes[2].plot(risk_component, 'r',linewidth = 5,  label='Risk Component')
        axes[2].set_title(f'Risk Component ')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Value')

        plt.tight_layout()
        plt.show()





