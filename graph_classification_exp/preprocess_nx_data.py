import os
import pickle
import shutil

import networkx as nx
dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))
import argparse

def to_adj_str(adj):
    degree = adj.sum(axis = 1)
    node_num = adj.shape[0]
    str_cache = ''
    for i in range(node_num):
        neighbor = []
        for idx, edge in enumerate(adj[i].tolist()):
            if edge:
                neighbor.append(str(idx))
        str_cache +=  '0 ' + str(degree[i]) + ' ' + ' '.join(neighbor) + '\n'
    return str_cache

def save_to_txt(total_graph_num, str_data_list, model, dataset):
    if model == 'orig':
        folder_path = dir_path + '/dataset/{}/'.format(dataset)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        save_path = dir_path + '/dataset/{}/{}.txt'.format(dataset, dataset)
    else:
        folder_path = dir_path + '/dataset/{}_{}/'.format(model, dataset)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        save_path = dir_path + '/dataset/{}_{}/{}_{}.txt'.format(model, dataset, model, dataset)

    with open(save_path, 'w') as file:
        file.write(str(total_graph_num) + '\n')
        for data in str_data_list:
            adj_str = data[0]
            node_num = data[1]
            graph_label = int(data[2])-1
            file.write('{} {}\n'.format(node_num, graph_label))
            file.write(adj_str)


def convert_data(model, dataset):
    # 1.read data
    if model == 'orig':
        data_path = root_path + '/data/orig/{}.pkl'.format(dataset)
    else:
        data_path = root_path + '/data/generated/{}_{}.pkl'.format(model, dataset)
        
    with open(data_path, 'rb') as file:
        data_list = pickle.load(file)

    # 2.convert to text data
    # demo:
    # 1500
    # 7 0
    # 0 6 1 2 3 4 5 6
    # 0 6 0 2 3 4 5 6
    # 0 6 0 1 3 4 5 6
    # 0 6 0 1 2 4 5 6
    # 0 6 0 1 2 3 5 6
    # 0 6 0 1 2 3 4 6
    # 0 6 0 1 2 3 4 5

    total_graph_num = len(data_list)
    str_data_list = []
    for nx_graph in data_list:
        adj = nx.to_numpy_array(nx_graph)
        assert adj.shape[0] > 0
        label = nx_graph.graph['label']
        str_data_list.append((to_adj_str(adj.astype(int)), str(adj.shape[0]), str(label)))

    save_to_txt(total_graph_num, str_data_list, model, dataset)



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='new_dblp2', help='[new_IMDB_MULTI, new_dblp2]')
    parser.add_argument('--model', type=str, default='DPGGAN', help='[orig, DPGGAN, DPGVAE, GVAE, NetGAN, GraphRNN]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(dir_path)
    args = arg_parser()
    print(args)
    convert_data(args.model, args.dataset)
