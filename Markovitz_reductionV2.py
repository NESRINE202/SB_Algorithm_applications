#Imports 
import numpy as np 
import matplotlib as plt 
import ising_modelV2
import computingV2 
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
        
    def Pbinaire(self,bits,n): 
        p = np.zeros((n,n*(bits)))
        pp = np.array([2**i for i in range(bits)])
        # pp= np.ones(bits)

        for i in range(n):
            start = i*(bits)
            p[i,start:start+len(pp)] = pp
        return p/(2**bits)
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
            P = np.zeros(((fraction),n*(fraction - 1 )))
            for i in range((fraction)): 
                P[i,j*(fraction )+i] = 1
            return P 


    
        p = self.Pbinaire(self.fraction,self.n_asset)

        I = np.ones(self.n_asset)
        I1 = np.ones(self.n_asset*self.fraction)
       
        H = 0.5 *(p.T @ self.V @ p @I1 - self.Lamda1*p.T@self.Mu
                  +2*self.Lamda2*(1-self.n_asset)*p.T@I) 
        
        A = np.zeros((self.n_asset*self.fraction,self.n_asset*self.fraction))
        for i in range(self.n_asset*self.fraction): 
            for j in range(self.n_asset*self.fraction): 
                A[i,j] = np.sum(p[:,i])*np.sum(p[:,j])

        J = -0.5*(np.dot(np.transpose(p), np.dot(self.V ,p))-self.Lamda2*A)


        return H, J
    
    def forces(self,positions): 
        p = self.Pbinaire(self.fraction,self.n_asset)
        I1 = np.ones(self.n_asset)
        I2 = np.ones(self.n_asset*self.fraction)
        force = np.zeros((self.n_cond_init,self.fraction*self.n_asset))
        for i in range(self.n_cond_init):
            force[i,:] = (0.5* p.T @ self.V @ p @ positions[i,:]
                    + 0.5* p.T @ self.V @ p@ I2 - 0.5 * self.Lamda1* p.T@self.Mu 
                    - self.Lamda2*(1 - self.n_asset - 0.5*(p @ positions[i,:]).T @ I1)* (p.T@I1))

        return force
        
    
    def SB_optimization(self,step, iteration, n_cond_init,temperature_fluctuation,a): 
        H,J = self.Reduction_to_Ising()
        states,energies,path = computingV2.compute_single_instance(len(H), step,iteration, n_cond_init,J,H,temperature_fluctuation,a, savetofile=False)

        return states,energies
    


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
        S = states[40,:,-1,0]
        SF= np.zeros(len(S))
        for i in range(len(S)): 
            if S[i]>=0: 
                SF[i] = 1
            else: 
                SF[i] = -1
        Choices = (S+1)/2
        Weights = np.dot(self.Pbinaire(fraction,n_asset),Choices)
        return Weights





