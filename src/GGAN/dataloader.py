import os
import sys

import torch
import numpy as np
import networkx as nx
import scipy
import scipy.sparse as sp
import pickle

from src.utils import graph_to_adj_list

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))

multi_graph_dataset = set(['relabeled_dblp2', 'new_dblp2', 'dblp2','new_IMDB_MULTI', 'IMDB_MULTI', 'Resampled_IMDB_MULTI'])

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def generate_feature(adj, n_eigenvector):
    # Use eigenvector as feature
    adj_orig = adj.copy()
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)  # eliminate diag element
    adj_orig.eliminate_zeros()

    # n_eigenvector will be bounded by (0.65 * node_num , args.n_eigenvector)
    node_num = adj_orig.shape[0]
    if n_eigenvector > 0.65 * node_num:
        n_eigenvector = int(0.65 * node_num)

    # graph spectical transformation
    adj_ = sp.coo_matrix(adj_orig)
    adj_ = adj_orig + sp.eye(adj_.shape[0]) # add diag back
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
    _, adj_features = scipy.linalg.eigh(adj_normalized, eigvals=(node_num - n_eigenvector, node_num - 1))

    actual_feature_dim = n_eigenvector
    return adj_features, actual_feature_dim



class Single_Graph_Dataset:
    def __init__(self, dataset_str, n_eigenvector, graph_adj=None, label=None):
        self.dataset_str = dataset_str
        self.n_eigenvector = n_eigenvector
        self.graph_adj = graph_adj
        self.label=label
        self.load_data()

    def load_data(self):
        if self.dataset_str == 'karate':
            G = nx.karate_club_graph()
            adj = nx.to_scipy_sparse_matrix(G)
            features = torch.eye(adj.shape[0])
        elif self.dataset_str in multi_graph_dataset:
            adj = self.graph_adj
        else:
            print("dataset: {} is unkown, Single_Graph_Dataset.".format(self.dataset_str))
            sys.exit(1)

        if self.n_eigenvector is not None and self.n_eigenvector != 0:
            features, actual_feature_dim = generate_feature(adj, self.n_eigenvector)
            self.actual_feature_dim = actual_feature_dim


        adj_temp = adj.copy() # eliminate 0 --> adj_orig
        adj_temp = adj_temp - sp.dia_matrix((adj_temp.diagonal()[np.newaxis, :], [0]), shape=adj_temp.shape)
        adj_temp = adj_temp + sp.dia_matrix((np.ones(adj_temp.shape[0]), [0]), shape=adj_temp.shape)
        adj_temp.eliminate_zeros()
        self.adj = adj_temp
        self.features = features
        self.adj_list = graph_to_adj_list(self.adj)


# TODO: add graph label
class Multi_Graph_Dataset:
    def __init__(self, dataset_str, n_eigenvector):
        self.dataset_str = dataset_str
        self.n_eigenvector = n_eigenvector
        self.load_data()

    def load_data(self):
        graph_size_list = []
        dataset_list = []
        with open(root_path + '/data/orig/' + self.dataset_str + '.pkl', 'rb') as tf:
            graph_set = pickle.load(tf)
        for graph in graph_set:
            label = graph.graph['label']
            graph_size_list.append(graph.number_of_nodes())
            sp_adj_matrix = nx.to_scipy_sparse_matrix(graph)
            dataset_list.append(Single_Graph_Dataset(self.dataset_str, self.n_eigenvector, graph_adj=sp_adj_matrix, label=label) )
        self.datasets = dataset_list



# def pickle_load_data(dataset_str):
#     if dataset_str == 'IMDB':
#         with open(sys.path[1] + '/data/' + 'IMDB_MULTI' + '.pickle', 'rb') as tf:
#             graph_set = pkl.load(tf)
#
#     elif dataset_str == 'reddit_binary_nx2':
#         with open(sys.path[1] + '/data/' + 'reddit_nx2' + '.pickle', 'rb') as tf:
#             graph_set = pkl.load(tf)
#     else:
#         print("dataset: {} doesn't exist.".format(dataset_str))
#         sys.exit(1)
#     return graph_set