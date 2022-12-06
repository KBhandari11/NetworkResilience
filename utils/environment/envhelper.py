import random 
import numpy as np
import torch
import networkx as nx
from scipy.stats import entropy
from igraph import Graph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def gen_graph(cur_n, g_type,seed=None):
    random.seed(seed)
    if g_type == 'erdos_renyi':
        g = Graph.Erdos_Renyi(n=cur_n, p=random.uniform(0.10,0.15))
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=random.randint(2,4), p=random.uniform(0.01,0.05),seed = seed)
        g = Graph.from_networkx(g)
    elif g_type == 'small-world':
        g = nx.newman_watts_strogatz_graph(n=cur_n, k=random.randint(2,5), p=random.uniform(0.1,0.2),seed = seed)
        g = Graph.from_networkx(g)
    elif g_type == 'barabasi_albert':
        g = Graph.Barabasi(n=cur_n, m=random.randint(1,3))
    elif g_type == 'geometric':
        g = Graph.GSG(n=cur_n, radius=random.uniform(0.1,0.4))
    g.vs['name'] = range(cur_n)
    return g

    
def gen_new_graphs(graph_type,seed = None):
    random.seed(seed)
    np.random.seed(seed)
    a = np.random.choice(graph_type) if len(graph_type) != 1 else graph_type[0]
    number_nodes = random.randint(30,50)
    graph = gen_graph(number_nodes, a,seed)
    #graph =add_super_node(graph)
    active = 1
    graph.vs["active"] = active
    return graph    
  

def reset(graph):
    active = 1
    graph.vs["active"] = active
    return graph   

# Helper functions for game details.
'''def get_lcc(g):
    return max(g.connected_components(), key=len) '''    
def get_lcc(G):
    found = set()

    comps = []
    for v in G.vs.indices:
        if v not in found:
            connected = G.bfs(v)[0]
            found.update(connected)
            comps.append(connected)

    return max(comps, key=len)

def molloy_reed(g):
  degs = np.array(g.degree())
  #degs = all_degree
  #nonmax_lcc = list(set(g.vs.indices).difference(set(get_lcc(g))))
  #degs = np.delete(all_degree, np.array(nonmax_lcc, dtype=int))#for non max LCC
  #degs = np.delete(deg,-1)#for supernode
  k = degs.mean()
  k2 = np.mean(degs** 2)
  if k ==0:
    beta = 0
  else:
    beta = k2/k
  return beta

def global_feature(g): 
    M = g.ecount()
    N = g.vcount()
    degs = np.array(g.degree())
    k1 = degs.mean()
    k2 = np.mean(degs** 2)
    div = k2 - k1**2
    if k1 != 0:
        heterogeneity = div/k1
        density = (2*M)/(N*(N-1))
        resilience = k2/k1
        degs.sort()
        gini = np.sum(degs * (degs + 1))/(M*N) - (N+1)/N
        entrop = entropy(degs/M)/N
        transitivity = g.transitivity_undirected()
    else:
        heterogeneity = 0
        density = (2*M)/(N*(N-1))
        resilience = 0
        gini = 0
        entrop = 0
        transitivity = g.transitivity_undirected()
    global_properties = np.hstack((density,resilience,heterogeneity,gini,entrop,transitivity))
    #global_properties = np.hstack((density,resilience,heterogeneity))
    global_properties = torch.from_numpy(global_properties.astype(np.float32))#.to(device)
    return global_properties

def get_Ball(g,v,l,n):
    if l == 1:
        return [v]
    else:
        for i in g.neighbors(v):
            if not(i in n):
                a = get_Ball(g,i,l-1,n)
                if a == None:
                    n = list(set().union([i],n))
                else:
                    n = list(set().union(a,[i],n))
            #print('n',n)
        if v in n:
            return list(set(n)-set([v]))
        else:
            return n
         

def get_ci(g, l):
    ci = []
    degs = np.array(g.degree())
    #G_nx = g.to_networkx()
    for i in g.vs.indices:
        n = get_Ball(g,i,l,[i])
        j = np.sum(degs[n] - 1)
        ci.append((g.degree(i) - 1) * j)
    ci = np.array(ci)
    if np.std(ci) != 0:
        ci = (ci - np.mean(ci)) / np.std(ci)
    else:
        ci = (ci - np.mean(ci))
    return ci

'''
def get_ci(g, l):
    ci = []
    degs = np.array(g.degree())
    for i in g.vs.indices:
        n = [path[-1] for path in g.get_all_shortest_paths(i) if path and len(path) <= l]
        print(n)
        if i in n:
            n = n.remove(i)
        print(n)
        n = np.array(n)
        j = np.sum(degs[n] - 1)
        ci.append((g.degree(i) - 1) * j)
    return ci
'''

def get_centrality_features(g):
    degree_centrality = np.array(g.degree()) / (g.vcount() - 1)
    #precolation_centrality = list(nx.percolation_centrality(g,attribute='active').values())
    #closeness_centrality = list(nx.closeness_centrality(g).values())
    try:
        eigen_centrality = np.array(g.eigenvector_centrality())
    except:
        #ARPACKOptions.tol =  int(10e-2)
        #value = Graph.arpack_defaults.tol = int(10e-2)
        eigen_centrality = np.array(g.eigenvector_centrality())
    #clustering_coeff = np.array(g.transitivity_local_undirected())
    #core_num = np.array(g.coreness())
    pagerank = np.array(g.personalized_pagerank())
    ci = np.array(get_ci(g, 3))
    #active = np.array(g.nodes.data("active"))[:,1]
    #x = np.column_stack((degree_centrality,eigen_centrality,pagerank, ci ))
    x = np.column_stack((degree_centrality,eigen_centrality,pagerank))
    #x = ci.reshape(-1,1)
    return x

def features(g): 
    #actualGraph = g.g(np.arange(len(g)-1)) #for actual graph
    #x = get_centrality_features(actualGraph) #with supernode
    x = get_centrality_features(g)
    #x[:-1,:] =x_actual
    #x_normed = (x - np.mean(x)) / np.std(x) #Standardize features
    #active_nodes =  np.where(np.array(list(g.nodes(data="active")))[:,1] == 0)[0]
    #x_normed[active_nodes,:]=np.zeros(np.shape(x_normed)[1])
    x = torch.from_numpy(x.astype(np.float32))#.to(device)
    x_normed = x #(x - torch.mean(x)) / torch.std(x)

    return x_normed

def reduceddegree(g): 
    x = torch.FloatTensor(g.degree()).reshape((-1, 1)) - 1
    return x

def network_dismantle(board, init_lcc):
    """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
    all_nodes = np.array(board.vs["active"])
    active_nodes = np.where(all_nodes == 1)[0]
    largest_cc = len(get_lcc(board))
    cond = True if len(active_nodes) <= 2 or board.ecount() == 1  or (largest_cc/init_lcc) <= 0.1 else False
    return cond, largest_cc

def board_to_string(board):
    """Returns a string representation of the board."""
    return " ".join(str(f) for _, f in board.vs["active"])

def from_igraph(graph):
    edges = [edge.tuple for edge in graph.es]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edge_index.shape[0] != 0:
        edge_index = to_undirected(edge_index)
    data = {}
    #data["features"] = features(graph)
    x =  reduceddegree(graph) #features(graph)
    #data["reduceddegree"] = x#(x - torch.mean(x)) / torch.std(x)

    data["edge_index"] = edge_index.view(2, -1)
    data = Data.from_dict(data)
    '''
    xs = []
    for key in ("features", "reduceddegree"):
        x = data[key]
        x = x.view(-1, 1) if x.dim() <= 1 else x
        xs.append(x)
        del data[key]
    '''
    data.x = x#torch.cat(xs, dim=-1)
    return data