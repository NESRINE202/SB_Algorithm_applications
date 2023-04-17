import numpy as np
import matplotlib.pyplot as plt 
from ising import *


def trace_histo(y_value):
    plt.hist(y_value,bins=11)
    plt.show()

ising_model = IsingModel(n, pas, iteration,M,H,n_cond_init)
E = ising_model.simulate()

y_value =[]
for x in E:
    y_value.append(min(x))

y_value.sort()
y_value.pop()
y_value.pop()
y_value.pop()
ymin = min(y_value)
for i in range(len(y_value)):
    y_value[i] = abs((y_value[i]-ymin)/ymin)

trace_histo(y_value)
