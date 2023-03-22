import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
G.add_node(2)
G.add_edge(2,2,weight=5)
G.add_node(3)
G.add_edge(2,3,weight=1)
G.add_node(1)
G.add_edge(2,1,weight=1)
print(G.nodes(),G.edges())
G.add_edge(1,3)

nx.draw(G)
plt.show()
print(G)