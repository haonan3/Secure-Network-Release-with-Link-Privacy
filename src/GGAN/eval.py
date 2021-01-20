import os
import pickle

import scipy
import igraph
import networkx as nx
import numpy as np
import powerlaw
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt
plt.switch_backend('agg')


dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
                                                                                                               n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def symmetrize_and_without_self_loop(adj_orig):
    def symmetrize(a):
        # print("symmetrize A!")
        a = a + a.T
        sum_a = a - np.diag(a.diagonal())
        sum_a[sum_a >= 1] = 1
        sum_a[sum_a < 1] = 0
        return sum_a

    # input must be np.array not sparse matrix
    if scipy.sparse.issparse(adj_orig):
        adj_ = adj_orig.todense()
    else:
        adj_ = adj_orig

    # remove self_loop
    np.fill_diagonal(adj_, 0)
    adj_orig = symmetrize(adj_)

    G = nx.from_numpy_array(adj_)
    G.remove_nodes_from(list(nx.isolates(G)))
    adj = nx.to_numpy_array(G)
    return adj


def compute_graph_statistics(A_in, Z_obs=None):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.

    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """
    A = A_in.copy()


    # important restriction
    A = symmetrize_and_without_self_loop(A)

    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean

    # node number & edger number
    statistics['node_num'] = A_graph.number_of_nodes()
    statistics['edge_num'] = A_graph.number_of_edges()

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    # statistics['wedge_count'] = statistics_wedge_count(A)

    # claw count
    # statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    # statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    # statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Assortativity
    # statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)

    # Clustering coefficient
    # statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / statistics['claw_count']

    # Number of connected components
    # statistics['n_components'] = connected_components(A)[0]

    # if Z_obs is not None:
    #     # inter- and intra-community density
    #     intra, inter = statistics_cluster_props(A, Z_obs)
    #     statistics['intra_community_density'] = intra
    #     statistics['inter_community_density'] = inter

    statistics['cpl'] = statistics_compute_cpl(A)

    return statistics


def stat_eval(G):
    return compute_graph_statistics(nx.to_scipy_sparse_matrix(G).toarray())



def load_graphs(file_path):
    with open(file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


# load saved graphs and calculate avg of each metric
if __name__ == '__main__':
    print(root_path)
    dataset = 'imdb' # 'dblp'

    data_folder = root_path + '/data/'

    if dataset == 'imdb':
        orig = load_graphs(data_folder + 'orig/new_IMDB_MULTI.pkl')
        dpgraphgan_graph = load_graphs(data_folder + 'generated/DPGraphGAN_new_IMDB_MULTI.pkl')
        dpgraphvae_graph = load_graphs(data_folder + 'generated/DPGraphVAE_new_IMDB_MULTI.pkl')
        netgan_graph = load_graphs(data_folder + 'generated/NetGAN_new_IMDB_MULTI.pkl')
        graphrnn_graph = load_graphs(data_folder + 'generated/GraphRNN_new_IMDB_MULTI.pkl')
        graphvae_graph = load_graphs(data_folder + 'generated/GraphVAE_new_IMDB_MULTI.pkl')
        graphgan_graph = load_graphs(data_folder + 'generated/GraphGAN_new_IMDB_MULTI.pkl')
    else:
        orig = load_graphs(data_folder + 'orig/new_dblp2.pkl')
        dpgraphgan_graph = load_graphs(data_folder + 'generated/DPGraphGAN_new_dblp2.pkl')
        dpgraphvae_graph = load_graphs(data_folder + 'generated/DPGraphVAE_new_dblp2.pkl')
        netgan_graph = load_graphs(data_folder + 'generated/NetGAN_new_dblp2.pkl')
        graphrnn_graph = load_graphs(data_folder + 'generated/GraphRNN_new_dblp2.pkl')
        graphvae_graph = load_graphs(data_folder + 'generated/GraphVAE_new_dblp2.pkl')
        graphgan_graph = load_graphs(data_folder + 'generated/GraphGAN_new_dblp2.pkl')

    # # link density
    # upper_link_density = 0
    # lower_link_density = 1
    # for g in orig:
    #     num_nodes = len(g)
    #     full_edge_num = num_nodes**2
    #     actual_edge_num = g.number_of_edges()
    #     edge_density = actual_edge_num/full_edge_num
    #     if edge_density > upper_link_density:
    #         upper_link_density = edge_density
    #     if edge_density < lower_link_density:
    #         lower_link_density = edge_density
    # print("Upper:{}; Lower:{}".format(upper_link_density, lower_link_density))


    # graph stat
    print(len(orig))
    print(len(graphgan_graph))
    # LCC, triangle_count, cpl, gini, rel_edge_distr_entropy
    LCC_list = []
    TC_list = []
    CPL_list = []
    GINI_list = []
    REDE_list = []
    for g, generated_g in zip(orig, graphgan_graph):
        LCC_list.append(stat_eval(generated_g)['LCC'])
        TC_list.append(stat_eval(generated_g)['triangle_count'])
        CPL_list.append(stat_eval(generated_g)['cpl'])
        GINI_list.append(stat_eval(generated_g)['gini'])
        REDE_list.append(stat_eval(generated_g)['rel_edge_distr_entropy'])

    print("avg LCC:{}".format(sum(LCC_list)/len(LCC_list)))
    print("avg TC:{}".format(sum(TC_list) / len(TC_list)))
    print("avg CPL:{}".format(sum(CPL_list) / len(CPL_list)))
    print("avg GINI:{}".format(sum(GINI_list) / len(GINI_list)))
    print("avg REDE:{}".format(sum(REDE_list) / len(REDE_list)))

    # dblp
    # avg LCC:163.2173913043478
    # avg TC:643.9565217391304
    # avg CPL:3.6229098437512697
    # avg GINI:0.5010830435774121
    # avg REDE:0.9010612433648646
    # imdb
    # avg LCC:31.12
    # avg TC:1508.62
    # avg CPL:1.6412643551055979
    # avg GINI:0.17521018241531106
    # avg REDE:0.9630393055580962

