import numpy as np
import random as rd

n = 100
pas = 0.001
iteration = 10000
n_cond_init=n
M=np.zeros((n,n))

for i in range(n):
            for j in range(i+1):
                c = rd.random()*10
                M[i,j] = c
                M[j,i] = c
H=np.zeros(n)
for i in range(n):
    H[i]=rd.random()