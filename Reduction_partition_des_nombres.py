import numpy as np 


class Sac():
    def __init__(self,capacite,signe) -> None:
        self.capacite = capacite 
        self.signe = signe

class Item(): 
    def __init__(self,poids) -> None:
        self.poids = poids 

    def attribution(self,sac): 
        self.sac = sac 



# def main():



# if __name__ == "__main__":
#     main()