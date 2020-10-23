import os
import sys

import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy
import scipy.sparse as sp
import pickle

from utils import graph_to_adj_list

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def pickle_load_data(dataset_str):
    if dataset_str == 'IMDB':
        with open(sys.path[1] + '/data/' + 'IMDB_MULTI' + '.pickle', 'rb') as tf:
            graph_set = pkl.load(tf)
            # G = graph_set[0]
            # adj = nx.to_scipy_sparse_matrix(G)

    elif dataset_str == 'reddit_binary_nx2':
        with open(sys.path[1] + '/data/' + 'reddit_nx2' + '.pickle', 'rb') as tf:
            graph_set = pkl.load(tf)
            # G = graph_set[5]
            # adj = nx.to_scipy_sparse_matrix(G)
    else:
        print("dataset: {} doesn't exist.".format(dataset_str))
        sys.exit(1)
    return graph_set


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

    def preprocess_citeseer(self, test_idx_reorder, test_idx_range, tx):
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        return tx

    def load_data(self):
        if self.dataset_str == 'karate':
            G = nx.karate_club_graph()
            adj = nx.to_scipy_sparse_matrix(G)
            features = torch.eye(adj.shape[0])
        elif self.dataset_str in ['cora', 'citeseer']:
            # load the data: x, tx, allx, graph
            names = ['x', 'tx', 'allx', 'graph']
            objects = []
            for i in range(len(names)):
                '''
                fix Pickle incompatibility of numpy arrays between Python 2 and 3
                https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
                '''
                with open(root_path + "/data/ind.{}.{}".format(self.dataset_str, names[i]), 'rb') as rf:
                    u = pkl._Unpickler(rf)
                    u.encoding = 'latin1'
                    cur_data = u.load()
                    objects.append(cur_data)

            x, tx, allx, graph = tuple(objects)
            test_idx_reorder = parse_index_file(root_path + "/data/ind.{}.test.index".format(self.dataset_str))
            test_idx_range = np.sort(test_idx_reorder)
            tx = self.preprocess_citeseer(test_idx_reorder, test_idx_range, tx) if self.dataset_str=='citeseer' else tx

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            features = torch.FloatTensor(np.array(features.todense()))
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        elif self.dataset_str in ['relabeled_dblp2', 'new_dblp2', 'dblp2', 'Resampled_IMDB_MULTI', 'new_IMDB_MULTI', 'IMDB_MULTI']:
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