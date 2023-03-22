import numpy as np
import random as rd

n = 15
pas = 0.001
iteration = 10000
nbrsimulation=15
M=np.zeros((n,n))
for i in range(n):
            for j in range(i+1):
                c = rd.randint(0,9)
                M[i,j] = c
                M[j,i] = c
H=np.zeros(n)
for i in range(n):
    H[i]=rd.randint(0,1)

