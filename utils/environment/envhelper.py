import random 
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

def gen_graph(cur_n, g_type,seed=None):
    random.seed(seed)
    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=cur_n, p=random.uniform(0.15,0.20),seed = seed)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=random.randint(2,4), p=random.uniform(0.1,0.5),seed = seed)
    elif g_type == 'small-world':
        g = nx.newman_watts_strogatz_graph(n=cur_n, k=random.randint(2,5), p=random.uniform(0.1,0.2),seed = seed)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=cur_n, m=random.randint(2,5),seed = seed)
    elif g_type == 'geometric':
        g = nx.random_geometric_graph(cur_n, random.uniform(0.1,0.4),seed = seed)
    return g

def add_super_node(graph):
    x = len(graph)
    ebunch = [(i,x) for i in range(x)]
    graph.add_node(x)
    graph.add_edges_from(ebunch)
    return graph
    
def gen_new_graphs(graph_type,seed = None):
    random.seed(seed)
    np.random.seed(seed)
    a = np.random.choice(graph_type) if len(graph_type) !=1 else graph_type[0]
    number_nodes = random.randint(30,50)
    graph = gen_graph(number_nodes, a,seed)
    #graph =add_super_node(graph)
    active = 1
    nx.set_node_attributes(graph,active, "active")
    return graph    
  

def reset(graph):
    active = 1
    nx.set_node_attributes(graph,active, "active")
    return graph   

# Helper functions for game details.
def get_lcc(g):
    return len(max(nx.connected_components(g), key=len))
    '''subGraph = g.subgraph(np.arange(len(g)-1))
    return len(max(nx.connected_components(subGraph), key=len))#for supernode'''

def molloy_reed(g):
  all_degree =   np.array(g.degree())[:,1]
  #degs = all_degree
  nonmax_lcc = list(set(g.nodes()).difference(set(max(nx.connected_components(g), key=len))))
  degs = np.delete(all_degree,nonmax_lcc)#for non max LCC
  #degs = np.delete(deg,-1)#for supernode
  k = degs.mean()
  k2 = np.mean(degs** 2)
  if k ==0:
    beta = 0
  else:
    beta = k2/k
  return beta

def global_feature(g): 
    subGraph = g#g.subgraph(np.arange(len(g)-1)) #for supernode
    M = len(subGraph.edges())
    N = len(subGraph)
    degs =   np.array(subGraph.degree())[:,1]
    k1 = degs.mean()
    k2 = np.mean(degs** 2)
    div = k2 - k1**2
    if k1 != 0:
        heterogeneity = div/k1
        density = (2*M)/(N*(N-1))
        resilience = k2/k1
        sorted_degs = sorted(degs)
        gini = sum([(i+1) * sorted_degs[i] for i in range(N)])/(M*N) - (N+1)/N
        entrop = entropy(degs/M)/N
        transitivity = nx.algorithms.cluster.transitivity(subGraph)
    else:
        heterogeneity = 0
        density = (2*M)/(N*(N-1))
        resilience = 0
        gini = 0
        entrop = 0
        transitivity = nx.algorithms.cluster.transitivity(subGraph)
    global_properties = np.hstack((density,resilience,heterogeneity,gini,entrop,transitivity))
    #global_properties = np.hstack((density,resilience,heterogeneity))
    global_properties = torch.from_numpy(global_properties.astype(np.float32))#.to(device)
    return global_properties

def get_ci(g,l):
    ci = []
    degs =   np.array(g.degree())[:,1]
    for i in g.nodes():
        n = list(nx.single_source_shortest_path_length(g, i, cutoff=l))[0:]
        j = np.sum(degs[n]-1)
        ci.append((g.degree(i)-1)*j)
    return ci

def get_centrality_features(g):
    degree_centrality = list(nx.degree_centrality(g).values())
    #precolation_centrality = list(nx.percolation_centrality(g,attribute='active').values())
    #closeness_centrality = list(nx.closeness_centrality(g).values())
    eigen_centrality = list(nx.eigenvector_centrality(g,tol=1e-03).values())
    clustering_coeff = list(nx.clustering(g).values())
    core_num = list(nx.core_number(g).values())
    pagerank = list(nx.pagerank(g).values())
    ci = get_ci(g,3)
    #active = np.array(g.nodes.data("active"))[:,1]
    #x = np.column_stack((925egree_centrality,clustering_coeff,pagerank, core_num ))
    x = np.column_stack((degree_centrality,eigen_centrality,pagerank,clustering_coeff, core_num, ci ))
    #x = np.column_stack((degree_centrality,eigen_centrality,pagerank))
    return x

def features(g): 
    #actualGraph = g.subgraph(np.arange(len(g)-1)) #for actual graph
    #x = get_centrality_features(actualGraph) #with supernode
    x = get_centrality_features(g)
    #x[:-1,:] =x_actual
    scaler = StandardScaler()
    x_normed = scaler.fit_transform(x)#Standardize features
    #active_nodes =  np.where(np.array(list(g.nodes(data="active")))[:,1] == 0)[0]
    #x_normed[active_nodes,:]=np.zeros(np.shape(x_normed)[1])
    x = torch.from_numpy(x_normed.astype(np.float32))#.to(device)
    return x

def reduceddegree(g): 
    degs =   np.array(g.degree())[:,1]
    x = degs-1
    x = x.reshape(-1,1)
    x = torch.from_numpy(x.astype(np.float32))
    return x

def network_dismantle(board, init_lcc):
    """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
    all_nodes =  np.array(list(board.nodes(data="active")))[:,1]
    active_nodes =  np.where(all_nodes == 1)[0]
    largest_cc = get_lcc(board)
    cond = True if len(active_nodes) <= 2 or len(board.edges()) == 1  or (largest_cc/init_lcc) <= 0.1 else False
    return cond, largest_cc

def board_to_string(board):
    """Returns a string representation of the board."""
    value = np.array(list(board.nodes(data="active")))
    return " ".join(str(f) for e, f in value)