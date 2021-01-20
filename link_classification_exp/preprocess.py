import argparse
import os
import pickle
import random
import shutil
import numpy as np
import networkx as nx

from src.dataloader import Single_Graph_Dataset

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))


test_edge_ratio = 0.3


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43, help='Random seed.')

    args = parser.parse_args()
    return args



def process_orig_graph(graph_list, Gs):
    test_edges_list = []
    test_edges_list_neg = []
    train_edges_list = []
    node_num_list = []
    for i, graph_edges in enumerate(graph_list):
        node_num_list.append(Gs[i].number_of_nodes())
        num_edge = len(graph_edges)
        hold_out_num = int(test_edge_ratio * num_edge)
        test_edge_idx = random.sample(range(num_edge), hold_out_num)
        train_edge_idx = list(set(list(range(num_edge))) - set(test_edge_idx))
        # train_edge_idx = list(set(list(range(num_edge))))
        assert hold_out_num > 0
        train_edges_list.append(list((np.array(graph_edges)[train_edge_idx])))
        test_edges_list.append(list((np.array(graph_edges)[test_edge_idx])))

        not_exist_edge = []
        for non_edge in nx.non_edges(Gs[i]):
            not_exist_edge.append([int(non_edge[0]), int(non_edge[1])])
        test_edge_idx_neg = random.sample(range(len(not_exist_edge)), hold_out_num)
        test_edges_list_neg.append(list((np.array(not_exist_edge)[test_edge_idx_neg])))

    return train_edges_list, test_edges_list, test_edges_list_neg, node_num_list



def process_generated_graph(graphs_list, test_edges_list):
    gen_train_edges_list = []
    assert len(graph_list) == len(test_edges_list)
    for idx, edge_list in enumerate(graphs_list):
        test_edge_list = test_edges_list[idx]
        test_edge_tuple = [tuple(x) for x in test_edge_list]
        edge_tuple = [tuple(x) for x in edge_list]
        gen_train_edge_list = list(set(edge_tuple) - set(test_edge_tuple))
        # gen_train_edge_list = list(set(edge_tuple))
        gen_train_edges_list.append(gen_train_edge_list)
    return gen_train_edges_list


def save_edge_list(node_num, file_name, edge_list):
    with open(file_name, 'w') as file:
        file.write(str(node_num) + '\n')
        for edge in edge_list:
            file.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')



def save_processed_graph(node_num_list, train_edges_list, test_edges_list=None, graph_dataset_name=None):
    # check folder exist or not
    folder_path = dir_path + '/dataset/{}/'.format(graph_dataset_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    for idx, edge_list in enumerate(train_edges_list):
        save_edge_list(node_num_list[idx], folder_path + 'train_edge_list_{}.txt'.format(idx), edge_list)
    if test_edges_list is not None:
        for idx, (test_edges_list_pos, test_edges_list_neg) in enumerate(test_edges_list):
            with open(folder_path + 'test_edge_list_{}.txt'.format(idx), 'w') as file:
                for edge in test_edges_list_pos:
                    file.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + '1' + '\n')
                for edge in test_edges_list_neg:
                    file.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + '0' + '\n')



def read_graph_list(dataset_folder):
    dataset_list = []
    with open(root_path + '/data/' + dataset_folder + '.pkl', 'rb') as tf:
        graph_set = pickle.load(tf)
    for graph in graph_set:
        edge_list = []
        for line in nx.generate_edgelist(graph, data=False):
            edge_list.append([int(line.split(' ')[0]), int(line.split(' ')[1])])
        dataset_list.append(edge_list)
    return dataset_list, graph_set


if __name__ == '__main__':
    args = arg_parser()
    for graph_set_name in ['new_dblp2', 'new_IMDB_MULTI']:
        orig_graph_list, Gs = read_graph_list('orig/{}'.format(graph_set_name))
        train_edges_list, test_edges_list, test_edges_list_neg, node_num_list = process_orig_graph(orig_graph_list, Gs)
        save_processed_graph(node_num_list, train_edges_list, list(zip(test_edges_list, test_edges_list_neg)), graph_set_name)
        # for model in ['GGAN_{}', 'DPGGAN_{}_eps:0.1', 'DPGGAN_{}_eps:1.0', 'DPGGAN_{}_eps:10.0']:
        for model in ['GVAE_{}']:
            graph_list, _ = read_graph_list('generated/{}'.format(model.format(graph_set_name)))
            gen_train_edges_list = process_generated_graph(graph_list, test_edges_list)
            save_processed_graph(node_num_list, gen_train_edges_list, test_edges_list=None, graph_dataset_name=model.format(graph_set_name))