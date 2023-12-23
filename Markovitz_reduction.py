#Imports 

import numpy as np 
import matplotlib as plt 
import ising_model
import computing 
import results_annalysis

class Markovitz:
    #In here the step , iteration , n_cond_init, fraction  are to optimize later  
    def __init__(self,fraction,V, Mu, Lamda1, Lamda2,Lamda, step, iteration, n_cond_init,temperature_fluctuation,a) -> None:
        #V is the covariance matrix 
        # Mu is the expected return vector 
        #Lambda 1 is the lagrange parametre for the expected return constraint 
        #Lamda 2 is the lagrange parameter for the constarint of the weight vector 
        #Lamda is a vector of lamdas that corresponds to the condition that only one 
        #n_asset is the number of assets in the portfolio 


        # Parameters of the POtrolio problem
        self.n_asset = len(Mu)
        self.fraction = fraction 
        self.V = V 
        self.Mu = Mu 
        self.Lamda1 = Lamda1
        self.Lamda2 = Lamda2
        self.Lamda = Lamda
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
        # This fuction guves us a projection matric able to respect 
        # the constraint of the sum equals to one fpr each asset 
        def Projection(j,n,fraction): 
            P = np.zeros(((fraction-1),n*(fraction - 1 )))
            for i in range((fraction - 1)): 
                P[i,j*(fraction - 1)+i] = 1
            return P 



        # Sum constraints of the spins of each asset 
                
        Hs = 0 
        e1 = np.ones((self.fraction - 1) )
        for i in range(self.n_asset): 
            Hs+= self.Lamda[i] * np.dot(np.transpose(e1), Projection(i,self.n_asset,self.fraction))



        
    
        p = P(self.fraction,self.n_asset)

        I = np.ones(self.n_asset)
        H = (np.dot(np.transpose(p),np.dot(self.V,I)) 
             -self.Lamda1* np.dot(np.transpose(p),self.Mu)
             + self.Lamda2*np.dot(np.transpose(p),I) )/2 + Hs
        
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









    



    

    


