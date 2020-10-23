import argparse
import pickle
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
import itertools


def read_graph(data_path):
    with open(data_path, 'rb') as tf:
        graph_set = pickle.load(tf)
    return graph_set

# get total node, total edge, avg node per graph, avg edge per graph
def data_stat(graph_set):
    graph_num = 0
    total_node = 0
    total_edge = 0
    for graph in graph_set:
        total_node += graph.number_of_nodes()
        total_edge += graph.number_of_edges()
        graph_num += 1
    print('Total graph Number:{}, total node:{}, total edge:{}, avg node:{}, avg edge:{}'.
          format(graph_num, total_node, total_edge, (total_node/graph_num), (total_edge/graph_num)))


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

'''
Measure pairwise graph stat distribution. If the generated graph has similar distribution respect to orig graph,
then we say the generated graph capture the essential of the orig graph.
'''


def get_degree_vec_list(graph_list):
    degree_vec_list = []
    max_degree = 50
    for graph in graph_list:
        temp = np.array(list(graph.degree))[:, 1]
        temp[temp > max_degree] = max_degree
        counter = np.zeros((max_degree,))
        for i in range(max(temp)):
            counter[i] += (temp == i).sum()
        degree_vec_list.append(counter)
    return degree_vec_list


def load_graphs(file_path):
    with open(file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def cosine_similarity(orig, g):
    total_score = 0
    assert len(orig) == len(g)
    for i in range(len(orig)):
        orig_vec = torch.tensor(orig[i]).reshape(1, -1)
        g_vec = torch.tensor(g[i]).reshape(1, -1)
        orig_vec = F.normalize(orig_vec, dim=-1, p=2)
        g_vec = F.normalize(g_vec, dim=-1, p=2)
        score = (orig_vec * g_vec).sum()
        total_score += score
    return total_score / len(orig)


def main_degree_distribution_similarity(orig, dpgraphgan_graph, dpgraphvae_graph, netgan_graph, graphrnn_graph, graphvae_graph):
    orig_degree_vec_list = get_degree_vec_list(orig)
    netgan_degree_vec_list = get_degree_vec_list(netgan_graph)
    graphrnn_degree_vec_list = get_degree_vec_list(graphrnn_graph)
    graphvae_degree_vec_list = get_degree_vec_list(graphvae_graph)
    dpgraphgan_degree_vec_list = get_degree_vec_list(dpgraphgan_graph)
    dpgraphvae_degree_vec_list = get_degree_vec_list(dpgraphvae_graph)
    
    generated_graph_list = [graphvae_degree_vec_list, netgan_degree_vec_list, graphrnn_degree_vec_list,\
                            dpgraphvae_degree_vec_list, dpgraphgan_degree_vec_list]
    for g in generated_graph_list:
        degree_similarity_score = cosine_similarity(orig_degree_vec_list, g)
        print(degree_similarity_score)


def get_motif_array(graph_list):
    motif_list = hard_code_motif()
    motif_array = np.zeros((len(graph_list), len(motif_list)))
    for g_idx, g in enumerate(graph_list):
        for m_idx, motif in enumerate(motif_list):
            print("{}-{}".format(g_idx, m_idx))
            for sub_nodes in itertools.combinations(g.nodes(), len(motif.nodes())):
                subg = g.subgraph(sub_nodes)
                if nx.is_connected(subg) and nx.is_isomorphic(subg, motif):
                    motif_array[g_idx, m_idx] += 1
    return motif_array


def main_motif_distribution_similarity(orig, dpgraphgan_graph, dpgraphvae_graph, netgan_graph, graphrnn_graph, graphvae_graph):
    orig_motif_array = get_motif_array(orig)
    graphvae_motif_array = get_motif_array(graphvae_graph)
    netgan_motif_array = get_motif_array(netgan_graph)
    graphrnn_motif_array = get_motif_array(graphrnn_graph)
    dpgraphvae_motif_array = get_motif_array(dpgraphvae_graph)
    dpgraphgan_motif_array = get_motif_array(dpgraphgan_graph)
    
    result_list = []
    generated_graph_list = [graphvae_motif_array, netgan_motif_array, graphrnn_motif_array,\
                            dpgraphvae_motif_array, dpgraphgan_motif_array]
    for g in generated_graph_list:
        motif_similarity_score = cosine_similarity(orig_motif_array, g)
        result_list.append(motif_similarity_score)
        print(motif_similarity_score)
    return result_list


def main_motif_distribution_similarity_one_graph(orig, generated_graph, top_N_graph=None):
    if top_N_graph is not None:
        orig_motif_array = get_motif_array(orig[:top_N_graph])
        generated_motif_array = get_motif_array(generated_graph[:top_N_graph])
    else:
        orig_motif_array = get_motif_array(orig)
        generated_motif_array = get_motif_array(generated_graph)
    motif_similarity_score = cosine_similarity(orig_motif_array, generated_motif_array)
    return [motif_similarity_score]


def motif_test():
    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 7)
    g.add_edge(2, 4)
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(3, 6)
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(6, 7)
    
    target = nx.Graph()
    target.add_edge(1, 2)
    target.add_edge(2, 3)
    
    for sub_nodes in itertools.combinations(g.nodes(), len(target.nodes())):
        subg = g.subgraph(sub_nodes)
        if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
            print(subg.edges())


def hard_code_motif():
    # hard code all motifs from 3 nodes to 5 nodes
    # there are 29 motifs from 3 nodes to 5 nodes
    motif_list = []
    
    g1 = nx.Graph()
    g1.add_edge(1, 2)
    g1.add_edge(1, 3)
    motif_list.append(g1)
    
    g2 = nx.Graph()
    g2.add_edge(1, 2)
    g2.add_edge(1, 3)
    g2.add_edge(2, 3)
    motif_list.append(g2)
    
    g3 = nx.Graph()
    g3.add_edge(1, 2)
    g3.add_edge(2, 3)
    g3.add_edge(3, 4)
    motif_list.append(g3)
    
    g4 = nx.Graph()
    g4.add_edge(1, 2)
    g4.add_edge(2, 3)
    g4.add_edge(2, 4)
    motif_list.append(g4)
    
    g5 = nx.Graph()
    g5.add_edge(1, 2)
    g5.add_edge(2, 3)
    g5.add_edge(3, 4)
    g5.add_edge(4, 1)
    motif_list.append(g5)
    
    g6 = nx.Graph()
    g6.add_edge(1, 2)
    g6.add_edge(2, 3)
    g6.add_edge(3, 1)
    g6.add_edge(4, 3)
    motif_list.append(g6)
    
    g7 = nx.Graph()
    g7.add_edge(1, 2)
    g7.add_edge(2, 3)
    g7.add_edge(3, 4)
    g7.add_edge(4, 1)
    g7.add_edge(1, 3)
    motif_list.append(g7)
    
    g8 = nx.Graph()
    g8.add_edge(1, 2)
    g8.add_edge(2, 3)
    g8.add_edge(3, 1)
    g8.add_edge(1, 4)
    g8.add_edge(2, 4)
    g8.add_edge(3, 4)
    motif_list.append(g8)
    
    g9 = nx.Graph()
    g9.add_edge(1, 2)
    g9.add_edge(2, 3)
    g9.add_edge(3, 4)
    g9.add_edge(4, 5)
    motif_list.append(g9)
    
    g10 = nx.Graph()
    g10.add_edge(1, 2)
    g10.add_edge(2, 3)
    g10.add_edge(3, 4)
    g10.add_edge(3, 5)
    motif_list.append(g10)
    
    g11 = nx.Graph()
    g11.add_edge(1, 2)
    g11.add_edge(2, 3)
    g11.add_edge(2, 4)
    g11.add_edge(2, 5)
    motif_list.append(g11)
    
    g12 = nx.Graph()
    g12.add_edge(1, 2)
    g12.add_edge(2, 3)
    g12.add_edge(3, 1)
    g12.add_edge(2, 4)
    g12.add_edge(3, 5)
    motif_list.append(g12)
    
    g13 = nx.Graph()
    g13.add_edge(1, 2)
    g13.add_edge(2, 3)
    g13.add_edge(3, 1)
    g13.add_edge(3, 4)
    g13.add_edge(4, 5)
    motif_list.append(g13)
    
    g14 = nx.Graph()
    g14.add_edge(1, 2)
    g14.add_edge(2, 3)
    g14.add_edge(3, 1)
    g14.add_edge(3, 4)
    g14.add_edge(3, 5)
    motif_list.append(g14)
    
    g15 = nx.Graph()
    g15.add_edge(1, 2)
    g15.add_edge(2, 3)
    g15.add_edge(3, 4)
    g15.add_edge(4, 5)
    g15.add_edge(1, 5)
    motif_list.append(g15)
    
    g16 = nx.Graph()
    g16.add_edge(1, 2)
    g16.add_edge(2, 3)
    g16.add_edge(3, 4)
    g16.add_edge(4, 1)
    g16.add_edge(4, 5)
    motif_list.append(g16)
    
    g17 = nx.Graph()
    g17.add_edge(1, 2)
    g17.add_edge(2, 3)
    g17.add_edge(3, 4)
    g17.add_edge(4, 1)
    g17.add_edge(1, 3)
    g17.add_edge(3, 5)
    motif_list.append(g17)
    
    g18 = nx.Graph()
    g18.add_edge(1, 2)
    g18.add_edge(2, 3)
    g18.add_edge(3, 1)
    g18.add_edge(3, 4)
    g18.add_edge(4, 5)
    g18.add_edge(3, 5)
    motif_list.append(g18)
    
    g19 = nx.Graph()
    g19.add_edge(1, 2)
    g19.add_edge(2, 3)
    g19.add_edge(3, 1)
    g19.add_edge(2, 4)
    g19.add_edge(3, 4)
    g19.add_edge(4, 5)
    motif_list.append(g19)
    
    g20 = nx.Graph()
    g20.add_edge(1, 2)
    g20.add_edge(1, 3)
    g20.add_edge(1, 4)
    g20.add_edge(2, 5)
    g20.add_edge(3, 5)
    g20.add_edge(4, 5)
    motif_list.append(g20)
    
    g21 = nx.Graph()
    g21.add_edge(1, 2)
    g21.add_edge(2, 3)
    g21.add_edge(3, 1)
    g21.add_edge(2, 4)
    g21.add_edge(3, 5)
    g21.add_edge(4, 5)
    motif_list.append(g21)
    
    g22 = nx.Graph()
    g22.add_edge(1, 2)
    g22.add_edge(2, 3)
    g22.add_edge(3, 4)
    g22.add_edge(4, 1)
    g22.add_edge(1, 3)
    g22.add_edge(1, 5)
    g22.add_edge(3, 5)
    motif_list.append(g22)
    
    g23 = nx.Graph()
    g23.add_edge(1, 2)
    g23.add_edge(2, 3)
    g23.add_edge(3, 1)
    g23.add_edge(1, 4)
    g23.add_edge(2, 4)
    g23.add_edge(3, 4)
    g23.add_edge(1, 5)
    motif_list.append(g23)
    
    g24 = nx.Graph()
    g24.add_edge(1, 2)
    g24.add_edge(2, 3)
    g24.add_edge(3, 4)
    g24.add_edge(4, 1)
    g24.add_edge(1, 3)
    g24.add_edge(1, 5)
    g24.add_edge(2, 5)
    motif_list.append(g24)
    
    g25 = nx.Graph()
    g25.add_edge(1, 2)
    g25.add_edge(1, 3)
    g25.add_edge(1, 4)
    g25.add_edge(2, 5)
    g25.add_edge(3, 5)
    g25.add_edge(4, 5)
    g25.add_edge(2, 3)
    motif_list.append(g25)
    
    g26 = nx.Graph()
    g26.add_edge(1, 2)
    g26.add_edge(2, 3)
    g26.add_edge(3, 4)
    g26.add_edge(4, 1)
    g26.add_edge(1, 3)
    g26.add_edge(1, 5)
    g26.add_edge(3, 5)
    g26.add_edge(2, 5)
    motif_list.append(g26)
    
    g27 = nx.Graph()
    g27.add_edge(1, 2)
    g27.add_edge(1, 3)
    g27.add_edge(1, 4)
    g27.add_edge(2, 5)
    g27.add_edge(3, 5)
    g27.add_edge(4, 5)
    g27.add_edge(2, 3)
    g27.add_edge(3, 4)
    motif_list.append(g27)
    
    g28 = nx.Graph()
    g28.add_edge(1, 2)
    g28.add_edge(2, 3)
    g28.add_edge(3, 4)
    g28.add_edge(4, 1)
    g28.add_edge(1, 3)
    g28.add_edge(1, 5)
    g28.add_edge(3, 5)
    g28.add_edge(2, 5)
    g28.add_edge(4, 5)
    motif_list.append(g28)
    
    g29 = nx.Graph()
    g29.add_edge(1, 2)
    g29.add_edge(1, 3)
    g29.add_edge(2, 3)
    g29.add_edge(2, 4)
    g29.add_edge(3, 4)
    g29.add_edge(2, 4)
    g29.add_edge(1, 5)
    g29.add_edge(2, 5)
    g29.add_edge(3, 5)
    g29.add_edge(4, 5)
    motif_list.append(g29)
    return motif_list



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imdb', help='[dblp, imdb]')
    parser.add_argument('--metric_type', type=str, default='degree', help='[degree, motif]')
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--top_N', type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    data_folder = parent_path + '/data/'
    log_folder = parent_path + '/log/txt_log/'
    args = arg_parser()
    if args.dataset == 'imdb':
        orig = load_graphs(data_folder + 'orig/new_IMDB_MULTI.pkl')
        dpgraphgan_graph = load_graphs(data_folder + 'generated/DPGraphGAN_new_IMDB_MULTI.pkl')
        dpgraphvae_graph = load_graphs(data_folder + 'generated/DPGraphVAE_new_IMDB_MULTI.pkl')
        netgan_graph = load_graphs(data_folder + 'generated/NetGAN_new_IMDB_MULTI.pkl')
        graphrnn_graph = load_graphs(data_folder + 'generated/GraphRNN_new_IMDB_MULTI.pkl')
        graphvae_graph = load_graphs(data_folder + 'generated/GraphVAE_new_IMDB_MULTI.pkl')
    else:
        orig = load_graphs(data_folder + 'orig/new_dblp2.pkl')
        dpgraphgan_graph = load_graphs(data_folder + 'generated/DPGraphGAN_new_dblp2.pkl')
        dpgraphvae_graph = load_graphs(data_folder + 'generated/DPGraphVAE_new_dblp2.pkl')
        netgan_graph = load_graphs(data_folder + 'generated/NetGAN_new_dblp2.pkl')
        graphrnn_graph = load_graphs(data_folder + 'generated/GraphRNN_new_dblp2.pkl')
        graphvae_graph = load_graphs(data_folder + 'generated/GraphVAE_new_dblp2.pkl')

    
    if args.metric_type == 'degree':
        main_degree_distribution_similarity(orig, dpgraphgan_graph, dpgraphvae_graph, netgan_graph, graphrnn_graph, graphvae_graph)
    else:
        if args.graph is None:
            result_list = main_motif_distribution_similarity(orig, dpgraphgan_graph, dpgraphvae_graph, netgan_graph, graphrnn_graph, graphvae_graph)
            args.graph = 'all'
        elif args.graph == 'dpgraphgan_graph':
            result_list = main_motif_distribution_similarity_one_graph(orig, dpgraphgan_graph, top_N_graph=args.top_N)
        elif args.graph == 'dpgraphvae_graph':
            result_list = main_motif_distribution_similarity_one_graph(orig, dpgraphvae_graph, top_N_graph=args.top_N)
        elif args.graph == 'netgan_graph':
            result_list = main_motif_distribution_similarity_one_graph(orig, netgan_graph, top_N_graph=args.top_N)
        elif args.graph == 'graphrnn_graph':
            result_list = main_motif_distribution_similarity_one_graph(orig, graphrnn_graph, top_N_graph=args.top_N)
        elif args.graph == 'graphvae_graph':
            result_list = main_motif_distribution_similarity_one_graph(orig, graphvae_graph, top_N_graph=args.top_N)
        else:
            print("wrong args.")
            exit(1)
            
        with open(log_folder+'{}_motif.txt'.format(args.dataset), 'a') as file:
            for result in result_list:
                file.write(args.graph + ',' + str(result) + '\n')