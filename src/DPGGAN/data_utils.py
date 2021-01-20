import pickle as pkl
import scipy
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

def load_data(dataset, n_eigenvector=None):
    adj, features = None, None

    # Set default feature_dim
    if n_eigenvector != None:
        feature_dim = n_eigenvector
    else:
        feature_dim = 15

    # Load data from specific dataset
    if dataset == 'karate':
        G = nx.karate_club_graph()
        adj = nx.to_scipy_sparse_matrix(G)
    elif dataset == 'cora' or dataset == 'citeseer':
        # load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            '''
            fix Pickle incompatibility of numpy arrays between Python 2 and 3
            https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
            '''
            with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
                u = pkl._Unpickler(rf)
                u.encoding = 'latin1'
                cur_data = u.load()
                objects.append(cur_data)
            # objects.append(
            #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = torch.FloatTensor(np.array(features.todense()))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # If use eigenvector as feature
    if n_eigenvector != None:
        node_num = adj.shape[0]
        adj_ = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj_.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
        _, features = scipy.linalg.eigh(adj_normalized, eigvals=(node_num - feature_dim, node_num - 1))
        features = torch.FloatTensor(features)

    features_normalize = True
    if features_normalize:
        features = normalize(features)

    return adj, features

def normalize(x):
    import torch.nn.functional as F
    x_normed = F.normalize(x, p=2, dim=1)
    return x_normed


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    # TODO: prevent delete edge from node having only one edge
    node_with_one_edge = set()
    for i,v in enumerate(adj.sum(axis=1)):
        if v[0] == 0:
            print("Node %d without neigh" % i)
            sys.exit()
        elif v[0] == 1:
            node_with_one_edge.add(i)

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0] # edges-(5278, 2)
    qualified_edges = []
    for i in edges:
        if i[0]  not in node_with_one_edge and i[1] not in node_with_one_edge:
            qualified_edges.append(i)
    qualified_edges = np.array(qualified_edges)
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    edges_set = set()
    qualified_edges_set = set()
    for edge in edges:
        edges_set.add((edge[0],edge[1]))
    for edge in qualified_edges:
        qualified_edges_set.add((edge[0], edge[1]))
    diff_edges = []
    for t in (edges_set - qualified_edges_set):
        diff_edges.append([t[0],t[1]])
    diff_edges = np.array(diff_edges)

    qualified_edge_idx = list(range(qualified_edges.shape[0]))
    np.random.shuffle(qualified_edge_idx)
    val_edge_idx = qualified_edge_idx[:num_val]
    test_edge_idx = qualified_edge_idx[num_val:(num_val + num_test)]
    train_edge_idx = qualified_edge_idx[(num_val + num_test):]
    test_edges = qualified_edges[test_edge_idx]
    val_edges = qualified_edges[val_edge_idx]
    train_edges = qualified_edges[train_edge_idx]
    train_edges = np.vstack([train_edges,diff_edges])
    # Catch-up for those isolated nodes
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    isolated_nodes = set()
    for i,v in enumerate(adj_train.sum(axis=1)):
        if v[0] == 0:
            isolated_nodes.add(i)
    patch_edges = []
    for i in isolated_nodes:
        j = np.random.choice(np.nonzero(adj[i].toarray())[1])
        patch_edges.append([i,j])
    patch_edges = np.array(patch_edges)
    train_edges = np.vstack([train_edges, patch_edges])


    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        redundant_index = np.where(rows_close)[1] # redundant element's index of a
        return np.any(rows_close), redundant_index

    test_edges_false = []
    edges_all = sparse_to_tuple(adj)[0]
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all)[0]:
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false))[0]:
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false))[0]:
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges)[0]:
            continue
        if ismember([idx_j, idx_i], train_edges)[0]:
            continue
        if ismember([idx_i, idx_j], val_edges)[0]:
            continue
        if ismember([idx_j, idx_i], val_edges)[0]:
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false))[0]:
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false))[0]:
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)[0]
    assert ~ismember(val_edges_false, edges_all)[0]
    assert ~ismember(val_edges, test_edges)[0]
    if ismember(val_edges, train_edges)[0]:
        remove_index = ismember(val_edges, train_edges)[1]
        print("element need to remove from val: %d" % len(remove_index))
        val_edges = np.delete(val_edges,remove_index,0)
    if ismember(test_edges, train_edges):
        remove_index = ismember(test_edges, train_edges)[1]
        print("element need to remove from test: %d" % len(remove_index))
        test_edges = np.delete(test_edges, remove_index, 0)
    assert ~ismember(val_edges, train_edges)[0]
    assert ~ismember(test_edges, train_edges)[0]

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def make_adj_label(index, adj_matrix):
    adj_label = np.zeros((len(index),len(index)))
    for i,v in enumerate(index):
        adj_label[i] = adj_matrix[v,index].toarray()
    return adj_label


def draw_graph(adj, path, plot_name, circle):
    G = None
    if scipy.sparse.issparse(adj):
        G = nx.from_scipy_sparse_matrix(adj)
    else:
        G = nx.from_numpy_matrix(adj)

    options = {
        'node_color': 'black',
        'node_size': 5,
        'line_color': 'grey',
        'linewidths': 0.1,
        'width': 0.1,
    }

    if circle:
        node_list = sorted(G.degree, key=lambda x: x[1], reverse=True)
        node2order = {}
        for i, v in enumerate(node_list):
            node2order[v[0]] = i

        new_edge = []
        for i in G.edges():
            new_edge.append((node2order[i[0]], node2order[i[1]]))

        new_G = nx.Graph()
        new_G.add_nodes_from(range(len(node_list)))
        new_G.add_edges_from(new_edge)

        nx.draw_circular(new_G, with_labels=True)

    else:
        nx.draw(G, **options)
    plt.savefig(path + '/' + plot_name + ".png")
    plt.clf()