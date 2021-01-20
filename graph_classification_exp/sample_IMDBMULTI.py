import argparse
import os
import pickle
import random
from collections import defaultdict

import networkx as nx
dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def relabel_dblp2(data_list):
    # according to the method that we create dblp2 data, we can relabel dblp2 data to three classes
    # For (label<24) ->0; (24<label<48) ->1; (48<label<72) ->2
    for graph in data_list:
        if graph.graph['label'] < 24:
            graph.graph['label'] = 0
        elif graph.graph['label'] < 48:
            graph.graph['label'] = 1
        elif graph.graph['label'] < 72:
            graph.graph['label'] = 2
    return data_list
    

def read_graph(dataset):
    data_path = root_path + '/data/orig/{}.pkl'.format(dataset)
    with open(data_path, 'rb') as file:
        data_list = pickle.load(file)
    if dataset == 'dblp2':
        data_list = relabel_dblp2(data_list)
    return data_list


def save_graph(dataset, data_list):
    if dataset == 'IMDB_MULTI':
        save_path = root_path + '/data/orig/Resampled_IMDB_MULTI.pkl'
    elif dataset == 'dblp2':
        save_path = root_path + '/data/orig/relabeled_dblp2.pkl'
    else:
        save_path = None

    with open(save_path, 'wb') as file:
        pickle.dump(data_list, file)
    print("finish dump.")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp2', help='[dblp2, IMDB_MULTI]')
    parser.add_argument('--num_per_class', type=int, default=None, help='sample 200 graphs for imdb dataset.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    graphs_dict = defaultdict(list)
    data_list = read_graph(args.dataset)
    for graph in data_list:
        graphs_dict[graph.graph['label']].append(graph)
    
    if args.num_per_class is not None:
        sampled_graph_list = [] # random.sample(data_list, num)
        for label, g_list in graphs_dict.items():
            sampled_graph_list.extend(random.sample(g_list, args.num_per_class))
    else:
        sampled_graph_list = data_list
        
    save_graph(args.dataset, sampled_graph_list)
    label_dict = defaultdict(int)
    for graph in sampled_graph_list:
        label_dict[graph.graph['label']] += 1
    print(label_dict)