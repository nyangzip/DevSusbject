import pandas as pd
from networkx.algorithms import bipartite
import networkx as nx

test = pd.read_csv('./sp21_student_kaggle.csv')
prob = nx.Graph()
# Add nodes with the node attribute "bipartite"
prob.add_nodes_from(list(test.Name), bipartite=0)
prob.add_nodes_from(list(test.Kaggle), bipartite=1)
node_set = []
for ii in test.Name:
    for jj in test.Kaggle:
        node_set.append((ii,jj))
prob_edge = list(set(node_set)) # len : 11928
prob.add_edges_from(prob_edge)
nx.draw(prob) #,with_labels=True)

# https://towardsdatascience.com/visualizing-networks-in-python-d70f4cbeb259
# prob1
G = nx.from_pandas_edgelist(test,source='Name',target='Kaggle',edge_attr='Weight')
nx.draw(G)

# prob2
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G,nodelist=test.Kaggle,node_size=test.Weight,pos=pos)
labels = nx.get_edge_attributes(G,'Weight')
weights = nx.get_edge_attributes(G,'Weight')
nx.draw_networkx_edges(G,pos,edge_color=[e[2]['Weight'] for e in G.edges(data=True)],
        width=2, edge_cmap=plt.cm.Greys, style='-')
namelist = []
kagglelist = []
for ii, node in enumerate(G):
    if node[:5] == 'https':
        kagglelist.append(node)
    else:
        namelist.append(node)

nx.draw_networkx_nodes(G, pos,
                       nodelist=kagglelist,  # 노드 이름
                       node_color='#FF1744',  # 기본 'r', 'g', 'b' 색 지원
                       node_size=10)

nx.draw_networkx_nodes(G, pos,
                       nodelist=namelist,
                       node_color='#673AB7',
                       node_size=10)

# prob3
# Definition of component, https://frhyme.github.io/python-libs/nx_alg_connectivity/
nx.number_connected_components(G)
# node number 추가하기

# prob4
G_main = G.subgraph(sorted(nx.connected_components(G), key = len, reverse=True)[0])

# prob5
not_main_list = []
for ii in sorted(nx.connected_components(G), key = len, reverse=True)[1:]:
    not_main_list.append(sorted(list(ii))[0])

# prob6
bottom_nodes, top_nodes = bipartite.sets(G_main)

# prob7, visualize both persons graph and datasets graph using node-link diagrams
# 위에 이름 추가
cbg = nx.complete_bipartite_graph(bottom_nodes,top_nodes)
def bipartite_layout(inputG):
    ## bipartite한 graph의 layout
    if nx.is_bipartite(inputG) and nx.is_connected(inputG):## connected and bipartite
        bs1, bs2 = nx.bipartite.sets(inputG)
        pos = {}
        pos.update({n:(0, 1.0/(len(bs1)+1)*(i+1)) for i, n in enumerate(bs1)})
        pos.update({n:(1, 1.0/(len(bs2)+1)*(i+1)) for i, n in enumerate(bs2)})
        return pos
    else:# 이 경우 none을 리턴하므로, default layout으로 그림이 그려지게 됩니다. 
        print("it is not bipartite and not connected")

pos = bipartite_layout(cbg)

nx.draw_networkx_edges(G_main,pos,edge_color='gray',
        width=1, style='-')

# nx.draw_networkx(cbg, pos = pos)
nx.draw_networkx_nodes(bottom_nodes, pos,
                       node_color='#FF1744',  # 기본 'r', 'g', 'b' 색 지원
                       node_size=10)
nx.draw_networkx_nodes(top_nodes, pos,
                       node_color='#673AB7',  # 기본 'r', 'g', 'b' 색 지원
                       node_size=10)


# prob8, closeness centralities
a=nx.closeness_centrality(G_main)
a_list = []
list(a.keys())
for ii in list(a.keys()):
    if ii[:5] != 'https':
        print(f'Name: {ii} / Closeness: {a[ii]}')
        a_list.append(a[ii])

# prob9, betweeness centralities
b=nx.betweenness_centrality(G_main)
b_list = []
list(b.keys())
for ii in list(b.keys()):
    if ii[:5] != 'https':
        print(f'Name: {ii} / Betweeness: {b[ii]}')
        b_list.append(b[ii])

# prob10, correlation
import numpy as np
corrcoef=np.corrcoef(a_list,b_list)
corrcoef[0,1]
