import os
import pickle

import networkx as nx
import scipy
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))


class drawer:
    def __init__(self):
        pass

    def check_image_save_dir(self):
        pass
        # check the image save dir, according to the timestamp

    def save_graph(self):
        pass
        # save graph to relevant dir
        # write according to the following function.



def draw_graph(G, path, circle=False, color=None, remove_isolated=True):
    # G = None
    # if scipy.sparse.issparse(adj):
    #     G = nx.from_scipy_sparse_matrix(adj)
    # else:
    #     G = nx.from_numpy_matrix(adj)
    
    if remove_isolated:
        print("remove isolated nodes")
        G.remove_nodes_from(list(nx.isolates(G)))

    options = {
        'node_color': 'black',
        'node_size': 5,
        # 'line_color': 'grey',
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

    elif color is not None:
        nx.draw(G, node_color=color, node_size=20,  with_labels=True)
    else:
        nx.draw(G, **options)
    plt.savefig(path)
    plt.clf()
    
    
if __name__ == '__main__':
    print(root_path)
    # graph_list = ['GraphVAE_new_IMDB_MULTI', 'DPGraphVAE_new_IMDB_MULTI',
    #               'DPGraphGAN_new_IMDB_MULTI', 'GraphRNN_new_IMDB_MULTI', 'NetGAN_new_IMDB_MULTI',
    #               'GraphVAE_new_dblp2', 'DPGraphVAE_new_dblp2', 'DPGraphGAN_new_dblp2',
    #               'GraphRNN_new_dblp2', 'NetGAN_new_dblp2']
    graph_list = ['dblp2', 'new_IMDB_MULTI']
    ids = [0,1,2,3,4]
    for graph_name in graph_list:
        for id in ids:
            generated_graph_path = root_path + '/data/orig/{}.pkl'.format(graph_name)
            with open(generated_graph_path, 'rb') as file:
                graph_list = pickle.load(file)
            print(len(graph_list))
            save_path = root_path + '/data/graph_figures/orig_{}_{}.png'.format(graph_name, str(id))
            draw_graph(graph_list[id], path=save_path)