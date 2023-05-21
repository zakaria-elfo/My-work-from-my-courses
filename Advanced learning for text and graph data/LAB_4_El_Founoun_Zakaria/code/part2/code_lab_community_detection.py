"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################

    A = nx.adjacency_matrix(G)
    n = A.shape[0]

    D_inv = diags([1 /G.degree(node) for node in G.nodes()])
    L = np.eye(n) - D_inv @ A

    eign_val, eign_vect = eigs(L, which= 'SM', k = k)
    eign_val, eign_vect = np.real(eign_val), np.real(eign_vect)

    U = eign_vect.T
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U.T)

    clustering={x: kmeans.labels_[i] for i,x in enumerate(G.nodes())}

    return clustering


############## Task 7

##################
# your code here #
##################

G = nx.read_edgelist("../datasets/CA-HepTh.txt",delimiter="\t")
largest_cc = max(nx.connected_components(G), key=len)
largest_cc =  G.subgraph(largest_cc)
clustering_larg = spectral_clustering(largest_cc, 50)

############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    communities = set(clustering.values())
    n_c = len(communities)
    m = G.number_of_edges()
    modularity = 0
    for community in communities:
        nodes_c = [node for node in G.nodes() if clustering[node]== community]
        l_c = G.subgraph(nodes_c).number_of_edges()
        d_c = sum([G.degree(node) for node in nodes_c])
        modularity += l_c/m - (d_c/(2*m))**2
    
    return modularity



############## Task 9

##################
# your code here #
##################

random_clustering = {node: randint(0,49) for node in largest_cc.nodes()}

print("The spectral clustering modularity : ",modularity(largest_cc, clustering_larg))
print("The Random clustering modularity : ",modularity(largest_cc, random_clustering))






