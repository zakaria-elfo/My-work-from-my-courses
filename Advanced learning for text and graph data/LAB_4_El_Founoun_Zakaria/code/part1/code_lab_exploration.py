"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################
G = nx.read_edgelist("../datasets/CA-HepTh.txt", delimiter="\t",comments="#")

nodes_number = G.number_of_nodes()
edges_number = G.number_of_edges()
print(f"number of nodes : {nodes_number}") # number of nodes :  9877
print(f"number of edges : {edges_number}") # number of edges :  25998

############## Task 2

##################
# your code here #
##################
connected_components = nx.connected_components(G)
print(f"the number of connected components is {nx.number_connected_components(G)}")

largest_cc = max(connected_components, key = len)

nodes_cc = G.subgraph(largest_cc).number_of_nodes()
edges_cc = G.subgraph(largest_cc).number_of_edges()

print("The number of nodes of the largest component : ",nodes_cc)
print("The number of edges of the largest component : ",edges_cc)

print("The ratio of nodes of the largest component : ",nodes_cc/G.number_of_nodes())
print("the ratio of edges of the largest component : ",edges_cc/G.number_of_edges())



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################

print(f"the min degree is : {np.min(degree_sequence)}")
print(f"the max degree is : {np.max(degree_sequence)}")
print(f"the mean degree is : {np.mean(degree_sequence)}")
print(f"the median of degree is : {np.median(degree_sequence)}")


############## Task 4

##################
# your code here #
##################

frequencies = nx.degree_histogram(G)
plt.bar(np.arange(1, 66), frequencies[1:])
plt.title('The degree histogram')
plt.show()
#plt.savefig('The degree histogram.png')

# Now let's use log log scale 
fig, ax = plt.subplots(figsize = (20,20))
ax.bar(np.arange(1, 66), frequencies[1:])
ax.set_title('The degree histogram log-log scale')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
#plt.savefig('The degree histogram log-log scale')

############## Task 5

##################
# your code here #
##################

print(f"The global clustering coefficent is: {nx.transitivity(G)}")