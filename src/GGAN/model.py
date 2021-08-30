import math
import random
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from src.GGAN.aggregators import MeanAggregator
from src.GGAN.encoders import Encoder

# from gcn_layer import GraphConvolution
import numpy as np

from src.GGAN.gcn_layer import GraphConvolution


class GGAN(nn.Module):
    def __init__(self, args, model_args, features_np, adj_lists):
        super(GGAN,self).__init__()
        self.args = args
        self.features_np = features_np
        features = nn.Embedding(features_np.shape[0], features_np.shape[1])
        features.weight = nn.Parameter(torch.FloatTensor(features_np), requires_grad=False)
        # features.weight = nn.Parameter(torch.randn(features_np.shape[0], features_np.shape[1]), requires_grad=False)
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
        self.mapping1 = nn.Linear(self.dec2_dim, self.dec2_dim, bias=False)
        self.mapping2 = nn.Linear(self.dec2_dim, self.dec2_dim, bias=False)

        self.is_gan = 0
        if args.model_name == 'GGAN':
            self.disc_gcn = GraphConvolution(in_features=self.dec2_dim, out_features=self.dec2_dim)
            self.disc_linear = nn.Linear(in_features=self.dec2_dim, out_features=1, bias=False)
            self.is_gan = 1


    def discriminate(self, origin_feature, origin_adj, generated_adj):
        origin_embed = self.disc_gcn(origin_feature, origin_adj)
        orig_prob = self.disc_linear(origin_embed)
        pos_label = torch.ones_like(orig_prob)
        generated_embed = self.disc_gcn(origin_feature, generated_adj)
        generated_prob = self.disc_linear(generated_embed)
        neg_label = torch.zeros_like(generated_prob)
        pred = torch.cat((orig_prob, generated_prob), dim=0)
        label = torch.cat((pos_label, neg_label), dim=0)
        return pred, label


    def encode(self, nodes):
        _set = set
        _sample = random.sample
        node_num = len(self.adj_lists)

        # encode nodes by its neighs
        to_neighs = [self.adj_lists[int(node)] for node in nodes]

        # samp_neighs is neighs of nodes, self.enc2.num_sample == self.enc3.num_sample
        samp_neighs = [_set(_sample(to_neigh, self.enc2.num_sample, ))
                       if len(to_neigh) >= self.enc2.num_sample else to_neigh
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


    def decode(self, input_features):
        output = self.dec1(input_features)
        output = F.relu(output)
        output = self.dec2(output)
        emb1 = self.mapping1(output)
        emb2 = self.mapping2(output)
        return emb1, emb2

    def inner_product_with_mapping(self, emb1,emb2): # cos similarity
        emb1 = F.normalize(emb1, dim=-1, p=2)
        emb2 = F.normalize(emb2, dim=-1, p=2)
        adj = torch.mm(emb1, emb2.t())
        return adj

    def get_sub_adj_feat(self, nodes):
        subgraph_feature = []
        for i,v in enumerate(nodes):
            subgraph_feature.append(self.features_np[v])
        subgraph_feature_tensor = torch.FloatTensor(np.array(subgraph_feature))
        return subgraph_feature_tensor

    def forward(self, nodes, sub_adj, for_test=False):
        gan_pred, gan_label = None, None
        sub_adj_feat = self.get_sub_adj_feat(nodes)
        mu, logvar = self.encode(nodes)
        mu_q = F.normalize(mu, dim=-1, p=2)
        logvar_sub = -logvar
        std0 = self.args.std
        std_q = torch.exp(0.5 * logvar_sub) * std0
        epsilon = torch.randn(std_q.shape)
        h = mu_q + int(for_test) * epsilon * std_q
        emb1, emb2 = self.decode(h)
        reconst_adj = self.inner_product_with_mapping(emb1, emb2)

        if not for_test and self.is_gan:
            gan_pred, gan_label = self.discriminate(sub_adj_feat, sub_adj, reconst_adj)
        return mu, logvar_sub, reconst_adj, gan_pred, gan_label


    def loss_function(self, preds, labels, logvar_sub, pos_weight, gan_pred, gan_label):
        cost = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=torch.FloatTensor([pos_weight]))
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Trick: KL is constant w.r.t. to mu_q after we normalize mu_q.
        kl = (0.5 * (-logvar_sub + torch.exp(logvar_sub) - 1.0)).sum(dim=1).mean()
        if self.is_gan:
            gan_loss = F.binary_cross_entropy_with_logits(gan_pred, gan_label)
            return cost + self.args.kl_ratio * kl + self.args.discriminator_ratio * gan_loss
        else:
            return cost + self.args.kl_ratio * kl