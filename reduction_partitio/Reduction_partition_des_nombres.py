import numpy as np 

class Sac():
    def __init__(self,capacite,signe) -> None:
        self.capacite = capacite 
        self.signe = signe # correpond au code donne si on choisi le premier ou le deuxime sac
    def content_s(self,Items,signe):
        content =np.zeros(len(Items))
        for i in range(len(Items)):
            if Items[i]==signe:
                content[i] = i + 1 
        return content



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


    def reduction_f(self): 
        # construction de la Matrice J du modèle d'ising 
        n = self.Nombres_item
        J= np.zeros([n,n])
        for i in range(n): 
            for j in range(n): 
                if i != j : 
                    J[i,j]=self.Items[i]*self.Items[j]
                    J[j,i]= J[i,j]
                else:
                    J[i,i]=0
        return J 
    

print("nombres_Items = ")
Nombres_Items = 10
#definition of the items and building the matrix Items 
Items = np.zeros(Nombres_Items)
c = 0
for i in range(Nombres_Items):
    print("weight of Item ",i)
    weight=float(input())
    Items[i]=weight
    c += weight
C1 = c/2
C2=C1
Sac_1 = Sac(C1,1)
Sac_2 = Sac(C2,-1)
Sacs = [C1,C2]
reduction = Reduction(Nombres_Items,Items,Sacs)
J=reduction.reduction_f()
 


    

          