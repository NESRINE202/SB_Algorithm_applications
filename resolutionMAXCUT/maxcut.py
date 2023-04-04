import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import matrice


G=nx.Graph()
G.add_nodes_from(range(matrice.n))

edges=[]
for i in range(matrice.n):
    for j in range(i):
        edges.append([i,j,matrice.M[i,j]])


G.add_weighted_edges_from(edges)


############### faudra relier au solution trouver pour l'instant je prends un truc random
x=[1]*matrice.n
x[0]=-1
#########################################################################################



nodecolors = {i: "red" if x[i] == 1 else "green" for i in range(matrice.n)}

        
nx.draw(G, with_labels=True, node_color=[nodecolors[node] for node in G.nodes()])
plt.show()
    
        

