from matrice import M, H, n, pas, iteration, nbrsimulation
import numpy as np


def decimal_to_binary(k, n):
    binary = []
    while k > 0:
        binary = [2*(k % 2)-1] + binary
        k //= 2
    return (n - len(binary)) * [0] + binary


def calculeforce(M,t):
    return sum([M[i,j]*t[i]*t[j] for i in range(n) for j in range(n)])

best_t1 = [0] * n
min_value=calculeforce(M,best_t1)

for i in range(2 ** n):
    t1 = decimal_to_binary(i, n)
    m = np.dot(t1, np.dot(M, t1))
    if m < min_value:
        min_value = m
        best_t1 = t1

print(best_t1)
print(min_value)

    