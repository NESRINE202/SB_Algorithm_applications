import numpy as np
import random as rd
import matplotlib.pyplot as plt
import copy

from matrice import M,H,n,pas,iteration,n_cond_init



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
            
        f=-np.dot(M[i,:],X)-H[i]*X[i]
        
        return f

    
    def a(self, t): #a(t) dans le document thermal mais je n'ai aps encore compris l'utilite
        return 0 #fonction par hazard

    def temperature(self, t): #fonction pour donner la fluctuation du a la variation de temperature
        return 0.01

    # fonction de mise a jour des variables positions et vitesses
    def simplectic_update(self, X, V, t):

        forces = -np.dot(M, X)+H*X

        V = V * (1-self.a(t) + self.pas * self.temperature(t)) + self.pas * forces
        X = X + self.pas * V

        return X, V

    def signage(self, X):
        # donne le signe de chaque particule
        return [1 if X[i]>0 else -1 for i in range(self.n_part)]
    
    def simulate(self):
        
        for i in range(self.n_cond_init):
            
            position=np.array([rd.randint(0,1)*2-1 for i in range(self.n_part)]) # retourne une liste de 1 et -1 aleatoire
            vitesse=np.array([0]*self.n_part)
            
            E=[]
            for t in range(self.iteration):
                
                #calcule de l'energie de l'etat suivant
                E.append(self.calcule_energie(self.signage(position)))
                
                #calcule de l'etat suivant
                position, vitesse = self.simplectic_update(position, vitesse,t)
            
            
            ### affichage
            Y=np.arange(len(E))
            plt.plot(Y, E)
            print(self.signage(position))
        plt.title("niveau d'$é$n$é$rgie")    
        plt.xlabel("temps")
        plt.ylabel("E")
        plt.show()

        
# Utilisation de la classe IsingModel
print("je commence")

ising_model = IsingModel(n, pas, iteration,M,H,n_cond_init)
ising_model.simulate()
