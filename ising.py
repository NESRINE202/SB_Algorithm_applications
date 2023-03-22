import numpy as np
import random as rd
import matplotlib.pyplot as plt
import copy

from matrice import M,H,n,pas,iteration,nbrsimulation



class IsingModel:
    def __init__(self, n, pas, iteration,M,H,nbrsimulation):
        #n nombre de particule
        #M matrice des forces d'interactio,
        self.n = n
        self.pas = pas
        self.iteration = iteration
        self.M =M
        self.H=H
        self.nbrsimulation=nbrsimulation

    def calcule_energie(self, X):
        X1 = self.signage(X)
        return np.dot(X1, np.dot(self.M, X1))+np.dot(H,X1)
    
    
    def force(self, X, i):
        
        ####fonction pour limiter les amplitudes des X (paroi inelastique)
        def sign(x):
            criteremin=1 #on pourra changer
            if abs(x)<criteremin:
                return x/criteremin
            else:
                return int(x>0)-int(x<0)
        
        f = -sum([self.M[i,j]*sign(X[j] )for j in range(self.n)]+[self.H[i]*sign(X[i])])
        return f

    
    #cette fonction calcule l'etat suivant des variables
        
    def nouvelle_variable(self, X, vitesse, t): # X c'est position , t c'est le temp ou l'iteration
        
        def parametrecontrol(t): #a(t) dans le document thermal mais je n'ai aps encore compris l'utilite
            return 0 #fonction par hazard
        def temperature(t): #fonction pour donner la fluctuation du a la variation de temperature
            return 0.1
        
        # c'est pour calculer la nouvelle position de la particule
        X1 = copy.deepcopy(X)
        v = copy.deepcopy(vitesse)
        for i in range(self.n):
            v[i] = v[i]*(1-parametrecontrol(i)) + self.pas * (self.force(X, i) + temperature(t)*v[i])
            X1[i] = X1[i] + self.pas * v[i]
        return X1, v
    
    def signage(self, X):
        # donne le signe de chaque particule
        return [1 if X[i]>0 else -1 for i in range(self.n)]

    
       
    def simulate(self):
        
        for i in range(self.nbrsimulation):
            
            position=[rd.randint(0,1)*2-1 for i in range(self.n)] # retourne une liste de 1 et -1 aleatoire
            vitesse=[0]*self.n
            
            E=[]    
            for t in range(self.iteration):
                
                #calcule de l'energie de l'etat suivant
                E.append(self.calcule_energie(self.signage(position)))
                
                #calcule de l'etat suivant
                position, vitesse = self.nouvelle_variable(position, vitesse,t)
            
            
            ### affichage
            Y=[i for i in range(len(E))]
            plt.plot(Y, E)
            print(self.signage(position))
            
        plt.show()

        
# Utilisation de la classe IsingModel
print("je commence")

ising_model = IsingModel(n, pas, iteration,M,H,nbrsimulation)
ising_model.simulate()
