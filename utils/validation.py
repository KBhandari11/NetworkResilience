import networkx as nx
import numpy as np 
from sklearn.metrics import auc
from igraph import Graph

from utils.environment.envhelper import gen_new_graphs, reset

def crossvalidation_data_homogenity(validation_path):
    with open(validation_path, "rb") as fh:
        Cross_Graph = nx.read_edgelist(fh)

    nodes = Cross_Graph.nodes()
    NUM = len(nodes)
    map = {n: i for i, n in enumerate(nodes)}
    Cross_Graph = reset(Graph.from_networkx(nx.relabel_nodes(Cross_Graph, map)))
    x = np.flip(np.arange(NUM)[NUM:0:-1]/NUM)
    return Graph.from_networkx(Cross_Graph), x

def crossvalidation_data(graph_type=None,seed=None):
    if graph_type==None:
        Cross_Graph = gen_new_graphs(['erdos_renyi', 'powerlaw','small-world', 'barabasi_albert'],seed)
    else:   
        Cross_Graph = gen_new_graphs(graph_type,seed)
    NUM = Cross_Graph.vcount()
    x  =  np.flip(np.arange(NUM)[NUM:0:-1]/NUM)
    return Cross_Graph, x

def area_under_curve(number_nodes,lcc):
    return auc(number_nodes,lcc)

def get_Validation(num, file_path=None):
    evaluation = []
    x = []
    if file_path != None:
        for path in file_path:   
            for i in range(num):
                name = path+str(i)+".txt"
                graph, iteration =  crossvalidation_data_homogenity(name)
                evaluation.append(graph)
                x.append(iteration)
    else:
        for i in range(num):
            graph, iteration =  crossvalidation_data()
            evaluation.append(graph)
            x.append(iteration)
    return evaluation, x    
