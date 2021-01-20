import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, feature_dim, embed_dim, adj_lists, aggregator, first_layer,num_sample=10,  gcn=False, cuda=False):
        super(Encoder, self).__init__()
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.first_layer = first_layer

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        if self.gcn:
            output_dim = self.feat_dim
        else:
            output_dim = 2 * self.feat_dim
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, features, nodes, samp_neighs=None, feature_dict=None):
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
            if self.first_layer:
                self_feats = features(torch.LongTensor(nodes))
            else:
                self_feats = features[nodes]
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined.t()
