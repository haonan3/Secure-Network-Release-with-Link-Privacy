import argparse
import os
import random
from random import shuffle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as tdata
from tqdm import tqdm

from link_classification_exp.node2vec.src import node2vec
from link_classification_exp.node2vec.src.main import learn_embeddings

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))

new_dblp2_idx = [0,22]
new_IMDB_MULTI_idx = [0,99]
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--graph_name', type=str, default='new_dblp2', help="[new_dblp2, new_IMDB_MULTI]")
    parser.add_argument('--graph_type', type=str, default='{}', help="['{}', 'DPGGAN_{}_eps:0.1', 'DPGGAN_{}_eps:1.0', 'DPGGAN_{}_eps:10.0']")

    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_train_edgelist(data_folder_path, idx):
    train_file = '{}/train_edge_list_{}.txt'.format(data_folder_path, idx)
    edge_list = []
    node_num = None
    with open(train_file, 'r') as io:
        for idx, line in enumerate(io):
            if idx == 0:
                node_num = int(line.strip())
            else:
                line = line.strip()
                edge_list.append([int(line.split(' ')[0]), int(line.split(' ')[1])])
    G = nx.Graph()
    G.add_nodes_from(list(range(node_num)))
    G.add_edges_from(edge_list)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G = G.to_undirected()
    return G



def read_test_edgelist(graph_name, idx):
    test_file = '{}/dataset/{}/test_edge_list_{}.txt'.format(dir_path, graph_name, idx)
    edge_list = []
    with open(test_file, 'r') as io:
        for line in io:
            edge_list.append([int(line.split(' ')[0]), int(line.split(' ')[1]), int(line.split(' ')[2])])
    return edge_list




def train_node2vec(nx_G, args, test_edge_array):

    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    w2v = learn_embeddings(args, walks)

    # get edge & non-exist edge list
    edge_list = []
    for line in nx.generate_edgelist(nx_G, data=False):
        edge_list.append([line.split(' ')[0], line.split(' ')[1], 1])
    pos_num = len(edge_list)
    neg_edge_list = []
    for line_idx, line in enumerate(nx.non_edges(nx_G)):
        neg_edge_list.append([str(line[0]), str(line[1]), 0])
    # balance neg and pos
    sampled_idx = random.sample(range(len(neg_edge_list)), min(int(pos_num*1.5), len(neg_edge_list)))
    sampled_neg_edge_list = list((np.array(neg_edge_list)[sampled_idx]))
    edge_list.extend(sampled_neg_edge_list)
    edge_array = np.array(edge_list)
    edge_array_idx = np.array(list(range(edge_array.shape[0])))
    pairs_dataloader = tdata.DataLoader(torch.from_numpy(edge_array_idx), batch_size=args.batch_size,
                                        shuffle=True)

    input = np.concatenate((w2v[edge_array[:,0]], w2v[edge_array[:,1]]), axis=1)
    input_tensor = torch.FloatTensor(input).to(args.cuda)
    label = torch.FloatTensor(edge_array[:, -1].astype(int)).to(args.cuda)

    # make sure test node has embedding
    valid_idx = []
    valid_set = set(w2v.index2word)
    for i, test_edge in enumerate(test_edge_array):
        if str(test_edge_array[i, 0]) in valid_set and str(test_edge_array[i, 1]) in valid_set:
            valid_idx.append(i)
    if not len(valid_idx) > 0:
        print('no valid test edge!')
        return -1
    test_edge_array = test_edge_array[valid_idx]
    test_input = np.concatenate((w2v[test_edge_array[:, 0].astype(str)], w2v[test_edge_array[:, 1].astype(str)]), axis=1)
    test_input_tensor = torch.FloatTensor(test_input).to(args.cuda)
    test_label = torch.FloatTensor(test_edge_array[:, -1].astype(int)).to(args.cuda)

    #### train mlp
    mlp = MLP(input_size=128*2, hidden_size=128).to(args.cuda)
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr)

    max_auc = 0
    for epoch in range(args.epochs):
        print("\r", '{}/{}'.format(epoch, args.epochs), end="", flush=True)
        mlp.train()
        for batch_pairs_idx in pairs_dataloader:
            batch_input = input_tensor[batch_pairs_idx]
            batch_label = label[batch_pairs_idx]
            optimizer.zero_grad()
            pred = mlp(batch_input)
            loss = mlp.get_loss(pred, batch_label)
            loss.backward()
            optimizer.step()

        auc_result = link_pred_test(mlp,test_input_tensor, test_label)
        if auc_result > max_auc:
            max_auc = auc_result

    return max_auc


def link_pred_test(mlp,test_input_tensor, test_label):
    with torch.no_grad():
        pred = torch.sigmoid(mlp(test_input_tensor))
        auc_score = mlp.get_auc_score(pred, test_label)
    return auc_score


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out

    def get_loss(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target.reshape(-1,1))
        return loss

    def get_auc_score(self, pred, target):
        if args.cuda:
            return roc_auc_score(target.to('cpu').numpy().astype(int), pred.detach().to('cpu').numpy().reshape(-1))
        else:
            return roc_auc_score(target.numpy().astype(int), pred.detach().numpy().reshape(-1))


if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)

    args = parse_args()
    args.cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph_name = args.graph_name
    graph_type = args.graph_type
    folder_name = graph_type.format(graph_name)
    data_folder_path = '{}/dataset/{}'.format(dir_path, folder_name)
    print(data_folder_path)

    idx_range = new_dblp2_idx if graph_name == 'new_dblp2' else new_IMDB_MULTI_idx
    max_auc_list = []
    for idx in tqdm(range(idx_range[0], idx_range[1])):
        train_graph = read_train_edgelist(data_folder_path, idx)
        test_edgelist = read_test_edgelist(graph_name, idx)
        max_auc = train_node2vec(train_graph, args, np.array(test_edgelist))
        if max_auc == -1:
            continue
        max_auc_list.append(max_auc)
    avg_auc = sum(max_auc_list) / len(max_auc_list)

    # write result to file
    log_path = '{}/log/{}.txt'.format(dir_path, folder_name)
    with open(log_path, 'a') as f:
        f.write('{}\t{}\n'.format(dt_string, avg_auc))
    print('{}\t{}\n'.format(dt_string, avg_auc))