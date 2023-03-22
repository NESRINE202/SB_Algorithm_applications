from matrice import n, pas, iteration, nbrsimulation
import numpy as np
from ising import M,H


def decimal_to_binary(k, n): #retourne une liste de longuer n (nombre de particule) avec des 1 et des -1 
    #k c'est l ordre de la configurationet on a 2**n configuration total 
    # c'est du binaire mais avc des -1 a la place des 0
    binary = []
    while k > 0:
        binary = [2*(k % 2)-1] + binary
        k //= 2
    return (n - len(binary)) * [-1] + binary


def calculeforce(M,X):# calcue l'energie avec une configuration donnéé en t
    return np.dot(X, np.dot(M, X))+np.dot(H,X)


# parcour tt les cas pour avoir le min

best_t1 = [0] * n
min_value=calculeforce(M,best_t1)


for i in range(2 ** n):
    t1 = decimal_to_binary(i, n)
    m = calculeforce(M,t1)
    if m < min_value:
        min_value = m
        best_t1 = t1

print(best_t1)
print(min_value)
print("M=",M)
print("H=",H)

    