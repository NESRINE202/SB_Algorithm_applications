#Imports 

import numpy as np 
import matplotlib as plt 
import ising_model
import computing 
import results_annalysis

class Markovitz:
    #In here the step , iteration , n_cond_init, fraction  are to optimize later  
    def __init__(self,fraction,V, Mu, Lamda1, Lamda2, step, iteration, n_cond_init,temperature_fluctuation,a) -> None:
        #V is the covariance matrix 
        # Mu is the expected return vector 
        #Lambda 1 is the lagrange parametre for the expected return constraint 
        #Lamda 2 is the lagrange parameter for the constarint of the weight vector 
        #n_asset is the number of assets in the portfolio 

        # Parameters of the POtrolio problem
        self.n_asset = len(Mu)
        self.fraction = fraction 
        self.V = V 
        self.Mu = Mu 
        self.Lamda1 = Lamda1
        self.Lamda2 = Lamda2
        # Prameters of the optimization 
        self.step = step 
        self.iteration = iteration 
        self.n_cond_init = n_cond_init
        self.temperature_fluctuation = temperature_fluctuation
        self. a = a 
        

    
    def Reduction_to_Ising(self):
        # This the projection matrix (It's not but it can be seen like that )
        def P(fraction,n): 
            p = np.zeros((n,n*(fraction-1)))
            pp = np.array([i+1 for i in range(fraction -1 )])
            for i in range(n):
                start = i*(fraction-1)
                p[i,start:start+len(pp)] = pp
            return p/fraction
        
        p = P(self.fraction,self.n_asset)

        I = np.ones(self.n_asset)
        H = (np.dot(np.transpose(p),np.dot(self.V,I)) 
             -self.Lamda1* np.dot(np.transpose(p),self.Mu)
             + self.Lamda2*np.dot(np.transpose(p),I) )/2
        
        J = np.dot(np.transpose(p), np.dot(self.V ,p))/4

        return H, J 
        
    
    def SB_optimization(self,step, iteration, n_cond_init,temperature_fluctuation,a): 
        H,J = self.Reduction_to_Ising()
        states,energies,path = computing.compute_single_instance(len(H), step,iteration, n_cond_init,J,H,temperature_fluctuation,a, savetofile=False)

        return states, energies

    def Ising_to_Portfolio(self,step, iteration, n_cond_init,temperature_fluctuation,a): 
        # This the projection matrix (It's not but it can be seen like that )
        def P(fraction,n): 
            p = np.zeros((n,n*(fraction-1)))
            pp = np.array([i+1 for i in range(fraction -1 )])
            for i in range(n):
                start = i*(fraction-1)
                p[i,start:start+len(pp)] = pp
            return p/fraction
        
        fraction = self.fraction
        n_asset = self.n_asset
        states, energies = self.SB_optimization(step, iteration, n_cond_init,temperature_fluctuation,a)
        S = states[0,:,-1,0]

        Choices = (S+1)/2
        Weights = np.dot(P(fraction,n_asset),Choices)
        return Weights









    



    

    


