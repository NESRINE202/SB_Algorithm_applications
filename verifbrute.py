from matrice import n, pas, iteration, nbrsimulation
import numpy as np
from ising import M,H
#import ising_copy


def decimal_to_binary(k, n):
    # Return a list of length n with 1's and -1's representing the binary digits of k
    binary_str = format(k, 'b').zfill(n)
    
    return [-1 if bit == '0' else 1 for bit in binary_str]



def calculeforce(M,X):# calcue l'energie avec une configuration donnéé en t
    return np.dot(X, np.dot(M, X))+np.dot(H,X)


# parcour tt les cas pour avoir le min

best_t1 = [-1] * n
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

    