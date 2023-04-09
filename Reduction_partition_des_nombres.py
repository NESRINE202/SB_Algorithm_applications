import numpy as np 
from ising import *

class Sac():
    def __init__(self,capacite,signe) -> None:
        self.capacite = capacite 
        self.signe = signe # correpond au code donne si on choisi le premier ou le deuxime sac

class Item(): 
    def __init__(self,poids) -> None:
        self.poids = poids 

    def attribution(self,sac): 
        self.sac = sac 


class Reduction(): 
    def __init__(self,Nombres_item,Items,Sacs) -> None:
        self.Nombres_item= Nombres_item
        self.Items = Items # an array that  the weight of the items
        self.Sacs = Sacs #un tableau qui contient la capcité de chaque sac 


    def reduction(self): 
        # construction de la Matrice J du modèle d'ising 
        n = len(self.Nombres_item)
        J= np.zeros(n,n)
        for i in range(n): 
            for j in range(n): 
                if i != j : 
                    J[i,j]=self.Items[i]*self.Items[j]
                    J[j,i]= J[i,j]
                else:
                    J[i,i]=0
        self.J = J 


    def partition(self): 
        #l'algorithme de bifuraction à implementer 
        repartition = IsingModel.simulate()
        # Je suppose ici que Simulate retourne un tableau d'un signe qui correspond a chacun des sac qui tient compte des deux sac 
        # on peut faire la somme des poids et prendre deux qui ont le mem poids 
    
        self.repartition = repartition 

                





# def main():



# if __name__ == "__main__":
     #main()    
          