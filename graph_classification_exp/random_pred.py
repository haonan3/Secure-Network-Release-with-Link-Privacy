import argparse
import pickle
from collections import defaultdict
import random

from sklearn.metrics import accuracy_score

from util import load_data, separate_data

numberList = [111,222,333,444,555]
print("random item from list is: ", random.choice(numberList))

def read_graph(data_path):
    with open(data_path, 'rb') as tf:
        graph_set = pickle.load(tf)
    return graph_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="Resampled_IMDB_MULTI",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', type=int, default=1,
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "log.txt", help='output file')
    args = parser.parse_args()

    random_acc_list = []
    for _ in range(20):
        train_label_list = []
        test_label_list = []
        pred_label_list = []

        graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
        train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
        for g in train_graphs:
            train_label_list.append(g.label)
        for g in test_graphs:
            test_label_list.append(g.label)

        for _ in test_graphs:
            pred_label_list.append(random.choice(test_label_list))

        random_acc = accuracy_score(test_label_list, pred_label_list)
        random_acc_list.append(random_acc)

    print("random prediction acc score:{}".format(sum(random_acc_list)/len(random_acc_list)))