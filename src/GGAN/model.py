import math
import random
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from GraphGen.aggregators import MeanAggregator
from GraphGen.encoders import Encoder

class GraphGen(nn.Module):

    def __init__(self, args, model_args, features_np, adj_lists):
        super(GraphGen,self).__init__()
        features = nn.Embedding(features_np.shape[0], features_np.shape[1])
        # features.weight = nn.Parameter(torch.FloatTensor(features_np), requires_grad=False)
        features.weight = nn.Parameter(torch.randn(features_np.shape[0], features_np.shape[1]), requires_grad=False)
        self.features = features
        self.node_num = features_np.shape[0]
        self.feature_dim = features_np.shape[1]
        self.adj_lists = adj_lists
        self.layer1_dim = model_args.layer1_dim
        self.layer2_dim = model_args.layer2_dim
        self.dec1_dim = model_args.dec1_dim
        self.dec2_dim = model_args.dec2_dim
        self.samp_num = model_args.samp_num


        # Follow the mode used in graphSAGE, the agg with 'gcn=False'
        # layer1
        self.agg1 = MeanAggregator(cuda=False,first_layer=True,gcn=False)
        self.enc1 = Encoder(self.feature_dim, self.layer1_dim, adj_lists, self.agg1, num_sample=self.samp_num,
                                                   gcn=True, cuda=False, first_layer=True)
        # layer2 * 2
        self.agg2 = MeanAggregator(cuda=False,first_layer=False,gcn=False)
        self.enc2 = Encoder(self.layer1_dim, self.layer2_dim, adj_lists, self.agg2, num_sample=self.samp_num,
                                                    gcn=True, cuda=False, first_layer=False)

        self.agg3 = MeanAggregator(cuda=False,first_layer=False,gcn=False)
        self.enc3 = Encoder(self.layer1_dim, self.layer2_dim, adj_lists, self.agg3, num_sample=self.samp_num,
                                                    gcn=True, cuda=False, first_layer=False)
        # decoder layer1 & layer2
        self.dec1 = nn.Linear(self.layer2_dim, self.dec1_dim, bias=True)
        self.dec2 = nn.Linear(self.dec1_dim, self.dec2_dim, bias=True)

        # mapping
        self.mapping1 = Parameter(torch.Tensor(self.dec2_dim, self.dec2_dim))
        self.mapping2 = Parameter(torch.Tensor(self.dec2_dim, self.dec2_dim))


    def init_params(self):
        nn.init.xavier_normal_(self.mapping1)
        nn.init.xavier_normal_(self.mapping2)


    def encode(self, nodes):
        _set = set
        _sample = random.sample
        node_num = len(self.adj_lists)

        # encode nodes by its neighs
        to_neighs = [self.adj_lists[int(node)] for node in nodes]

        # samp_neighs is neighs of nodes, self.enc2.num_sample == self.enc3.num_sample
        samp_neighs = [_set(_sample(to_neigh, self.enc2.num_sample, ))
                       if len(to_neigh) >= self.enc2.num_sample
                       else to_neigh
                       for to_neigh in to_neighs]

        # unique_nodes_list is all nodes required in layer2
        unique_nodes_list = list(set.union(*samp_neighs) | set(nodes))

        # encode unique_nodes_list in layer1
        embeds_layer1 = self.enc1(self.features, unique_nodes_list)

        # Look-up dict for layer1's embedding
        feature_dict = {}
        for i, v in enumerate(unique_nodes_list):
            feature_dict[v] = i
        features_embeds = embeds_layer1

        # feed Look-up dict and features_embeds into layer2
        nodes_idx = []
        for i, v in enumerate(nodes):
            nodes_idx.append(feature_dict[v])
        mu = self.enc2(features_embeds, nodes_idx, samp_neighs, feature_dict=feature_dict)
        logvar = self.enc3(features_embeds, nodes_idx, samp_neighs, feature_dict=feature_dict)

        return mu, logvar


    def decode(self, input_features, nodes, for_test=False):
        output = self.dec1(input_features)
        output = F.tanh(output)
        output = self.dec2(output)
        # return output
        emb1 = torch.mm(output, self.mapping1)
        emb2 = torch.mm(output, self.mapping2)
        return emb1, emb2


    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def inner_product_with_mapping(self, emb1,emb2, nodes):
        adj = torch.mm(emb1, emb2.t())
        adj_without_sigmoid = adj
        return adj_without_sigmoid


    def forward(self, nodes, sub_adj, for_test=False):
        mu, logvar = self.encode(nodes)
        h = self.reparameterize(mu, logvar)
        emb1, emb2 = self.decode(h, nodes, for_test)
        adj_without_sigmoid = self.inner_product_with_mapping(emb1, emb2, nodes)
        return mu, logvar, adj_without_sigmoid, None, None

    def loss_function(self, preds, labels, mu, logvar, n_nodes, pos_weight, epoch, orig_prob, generated_prob):
        cost = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=torch.FloatTensor([pos_weight]))
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = - (0.5 / n_nodes) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD

