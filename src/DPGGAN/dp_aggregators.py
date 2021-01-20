import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, cuda=False, gcn=False, first_layer=True):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.cuda = cuda
        self.gcn = gcn
        self.first_layer = first_layer
        
    def forward(self, features, nodes, to_neighs, num_sample=10, feature_dict=None):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None and self.first_layer:
            _sample = np.random.choice
            # samp_neighs = [_set(_sample(to_neigh, num_sample,))
            #                if len(to_neigh) >= num_sample
            #                else to_neigh for to_neigh in to_neighs]
            samp_neighs = [ _set(_sample(list(to_neigh), num_sample)) for to_neigh in to_neighs]

        else:
            samp_neighs =  [ _set(list(to_neigh)) for to_neigh in to_neighs]

        # if self.gcn:
        #     samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        # num_neigh = mask.sum(1, keepdim=True) + 1
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            if self.first_layer:
                embed_matrix = features(torch.LongTensor(unique_nodes_list))
            else:
                node_idx = []
                for i,v in enumerate(unique_nodes_list):
                    node_idx.append(feature_dict[v])
                embed_matrix = features[node_idx]
        to_feats = mask.mm(embed_matrix)
        return to_feats
