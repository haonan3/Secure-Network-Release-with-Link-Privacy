import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, feature_dim, embed_dim, adj_lists, aggregator, first_layer, samp_neighs=None,
            num_sample=5,  gcn=False, cuda=False, batch_size=None,
            feature_transform=False):
        super(Encoder, self).__init__()
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.first_layer = first_layer
        self.samp_neighs = samp_neighs
        # if base_model != None:
        #     self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        if self.gcn:
            input_dim = self.feat_dim
        else:
            input_dim = 2 * self.feat_dim
        self.weight = nn.Parameter(torch.FloatTensor(batch_size, input_dim, embed_dim))
        init.xavier_uniform_(self.weight)

        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)


    def forward(self, features, nodes, samp_neighs=None, feature_dict=None, for_test=False):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        if self.first_layer:
            neigh_feats = self.aggregator.forward(features, nodes,
                                                  [self.adj_lists[int(node)] for node in nodes],
                                                  self.num_sample)
        else:
            assert (samp_neighs != None)
            assert (feature_dict != None)
            neigh_feats = self.aggregator.forward(features, nodes, samp_neighs,
                                                  self.num_sample, feature_dict=feature_dict)
        if not self.gcn:
            if self.cuda:
                self_feats = features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        if for_test:
            combined = F.relu(torch.mm(combined, self.weight[0]))
        else:
            combined = combined.view(combined.shape[0],1,combined.shape[1])
            combined = F.relu(torch.bmm(combined, self.weight))
            combined = combined.view(combined.shape[0], combined.shape[2])  # (192,128)
        assert (len(list(combined.shape)) == 2)
        return combined

        # if self.first_layer:
        #     neigh_feats = self.aggregator.forward(features, nodes,
        #                                           [self.adj_lists[int(node)] for node in nodes],
        #                                           self.num_sample)
        # else:
        #     assert ~(samp_neighs == None)
        #     assert ~(feature_dict == None)
        #     neigh_feats = self.aggregator.forward(features, nodes, samp_neighs,
        #                                           self.num_sample, feature_dict=feature_dict)
        # if not self.gcn:
        #     if self.first_layer:
        #         self_feats = features(torch.LongTensor(nodes))
        #     else:
        #         self_feats = features[nodes]
        #     combined = torch.cat([self_feats, neigh_feats], dim=1)
        # else:
        #     combined = neigh_feats
        # combined = F.relu(self.weight.mm(combined.t()))
        # return combined.t()
