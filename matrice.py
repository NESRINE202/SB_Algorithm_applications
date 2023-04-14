import numpy as np
import random as rd

n = 20
pas = 0.001
iteration = 10000
n_cond_init=n
# M=np.zeros((n,n))

# for i in range(n):
#        for j in range(i+1):
#               c = rd.random()*10
#               M[i,j] = c
#               M[j,i] = c

def generate_instance():
       M = np.random.uniform(low=0, high=1, size=(n, n))
       H = np.zeros(n)

       return M, H

H=np.zeros(n)
# for i in range(n):
#     H[i]=rd.random()