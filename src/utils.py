from collections import defaultdict
import numpy as np
import networkx as nx
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt
from DPGraphGen.data_utils import make_adj_label

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def graph_to_adj_list(adj):
    # Sparse adj matrix to adj lists
    G = nx.from_scipy_sparse_matrix(adj)
    adj_lists = defaultdict(set)

    # Check isolated node before training
    for node, adjacencies in enumerate(G.adjacency()):
        if len(list(adjacencies[1].keys())) == 0:
            print("Node %d is isolated !!!" % node)
            assert False
        adj_lists[node] = set(list(adjacencies[1].keys()))

    return adj_lists


def save_top_n(adj, n, threshold=None):
    if threshold is None:
        il1 = np.tril_indices(adj.shape[0])
        adj[il1] = float("-infinity")
        index = adj.reshape((-1,)).argsort()[-n:]
        (row, col) = divmod(index, adj.shape[0])
        top_n = np.zeros_like(adj)
        top_n[row,col] = 1
        m = np.ones_like(adj)
        m[(top_n + top_n.T) == 0] = 0
        return m, 0
    else:
        # find neck_value
        adj_ = adj.copy()
        il1 = np.tril_indices(adj_.shape[0])
        adj_[il1] = float("-infinity")
        index = adj_.reshape((-1,)).argsort()[-n:]
        last_one = index[0]
        (row, col) = divmod(last_one, adj_.shape[0])
        neck_value = adj[row,col]
        # convert predict adj to adj(0,1)
        adj[adj >= threshold] = 1
        adj[adj < threshold] = 0
        return adj, neck_value


def save_edge_num(graph):
    graph = graph - sp.dia_matrix((graph.diagonal()[np.newaxis, :], [0]), shape=graph.shape)
    graph.eliminate_zeros()
    assert np.diag(graph.todense()).sum() == 0
    original_garph = nx.from_scipy_sparse_matrix(graph)
    n = original_garph.number_of_edges()
    return n



def sample_subgraph(args, node_num, dataset):
    index = np.random.choice(node_num, args.batch_size, replace=False)
    sub_adj = make_adj_label(index, dataset.adj)
    return index, sub_adj