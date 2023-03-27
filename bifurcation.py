import numpy as np 
import matplotlib.pyplot as plt
from matrice import M, n  

def p(t):
    return 0.01*t

def Tracage_bifurcation(delta, phi0, M, K, T, N):
    X = np.zeros((N+1,n)) # fix the shape of X array
    Y = np.zeros((N+1,n))
    h = T / (N+1)
     
    def calculate_sum_of_column(X, k):
        """Calculate the sum of the k-th line of X array."""
        return np.sum(X[k,:])
            

    for i in range(n):
        X[i, 0] = 0
        for k in range(N):
            X[k+1,i] = X[k,i] + h * delta * Y[k,i] 
            Y[k+1,i] = Y[k,i] - h * ((K * X[k,i]**2 - p(k*h) + delta) * X[k,i] + phi0 * calculate_sum_of_column(X,k))
    
    Temps = np.linspace(0, T, N+1)  # fix the linspace call
    P = [p(t) for t in Temps ]
    plt.plot(Temps,P)
    plt.plot(Temps, X[:,0])  # plot the first column of X
    plt.show()
    

#test pour x1
#il faut voir a quoi correspond les delta , k , phi 0 
N = 1000
delta = 0.5
phi0 = 0.1
K = 1
T=200
Tracage_bifurcation(delta,phi0,M,K,T,N)
    




