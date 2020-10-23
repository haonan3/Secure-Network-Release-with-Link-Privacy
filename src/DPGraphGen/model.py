import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from DPGraphGen.DPCounter import DPCounter
from DPGraphGen.dp_aggregators import MeanAggregator as DPMeanAggregator
from DPGraphGen.dp_encoders import Encoder as DPEncoder
from DPGraphGen.gcn_layer import GraphConvolution
from GraphGen.aggregators import MeanAggregator
from GraphGen.encoders import Encoder
import DPGraphGen.linear as linear
from DPGraphGen.utils_dp import create_cum_grads


class DPGraphGen(nn.Module):

    def __init__(self, args, model_args, features_np, adj_lists):
        super(DPGraphGen, self).__init__()
        self.features_np = features_np
        features = nn.Embedding(features_np.shape[0], features_np.shape[1])
        features.weight = nn.Parameter(torch.FloatTensor(features_np), requires_grad=False)
        # features.weight = nn.Parameter(torch.randn(features_np.shape[0], features_np.shape[1]), requires_grad=False)
        self.features = features
        self.disc_factor = args.discriminator_ratio
        self.node_num =  features_np.shape[0]
        self.feature_dim = features_np.shape[1]
        self.adj_lists = adj_lists
        self.layer1_dim = model_args.layer1_dim
        self.layer2_dim = model_args.layer2_dim
        self.dec1_dim = model_args.dec1_dim
        self.dec2_dim = model_args.dec2_dim
        self.samp_num = model_args.samp_num
        self.batch_size = args.batch_size
        self.batch_proc_size = model_args.batch_proc_size
        self.check_proc_size()
        self.dp_counter = DPCounter(args, model_args)
        assert (self.batch_size == self.batch_proc_size)


        # Follow the mode used in graphSAGE, the agg with 'gcn=False'
        # layer1
        self.agg1 = DPMeanAggregator(cuda=False,first_layer=True,gcn=False)
        self.enc1 = DPEncoder(self.feature_dim, self.layer1_dim,
                              self.adj_lists, self.agg1, num_sample=self.samp_num,
                            gcn=True, cuda=False, first_layer=True, batch_size=self.batch_size*(self.samp_num+1))
        self.bn1 = nn.BatchNorm1d(num_features=self.layer1_dim, affine=False)

        # layer2
        self.agg2 = DPMeanAggregator(cuda=False,first_layer=False,gcn=False)
        self.enc2 = DPEncoder(self.layer1_dim, self.layer2_dim,
                              self.adj_lists, self.agg2, num_sample=self.samp_num,
                            gcn=True, cuda=False, first_layer=False, batch_size=self.batch_size)
        self.bn2 = nn.BatchNorm1d(num_features=self.layer2_dim, affine=False)

        self.agg3 = DPMeanAggregator(cuda=False,first_layer=False,gcn=False)
        self.enc3 = DPEncoder(self.layer1_dim, self.layer2_dim,
                              self.adj_lists, self.agg3, num_sample=self.samp_num,
                            gcn=True, cuda=False, first_layer=False, batch_size=self.batch_size)
        self.bn3 = nn.BatchNorm1d(num_features=self.layer2_dim, affine=False)

        # decoder layer1
        self.dec1 = linear.Linear(self.layer2_dim, self.dec1_dim, bias=False, batch_size=self.batch_size)
        self.bn4 = nn.BatchNorm1d(num_features=self.dec1_dim, affine=False)
        # decoder layer2
        self.dec2 = linear.Linear(self.dec1_dim, self.dec2_dim, bias=False, batch_size=self.batch_size)
        self.bn5 = nn.BatchNorm1d(num_features=self.dec2_dim, affine=False)

        self.mapping1 = linear.Linear(self.dec2_dim, self.dec2_dim, bias=False, batch_size=self.batch_size)
        self.bn6 = nn.BatchNorm1d(num_features=self.dec2_dim, affine=False)

        # TODO: can we pass self as args into this func?
        self.cum_grads = create_cum_grads(self)

        # gan part
        self.disc_gcn = GraphConvolution(in_features=self.dec2_dim, out_features=self.dec2_dim)
        self.disc_linear = nn.Linear(in_features=self.dec2_dim, out_features=1, bias=False)


    def check_proc_size(self):
        self.batch_size = self.node_num if self.batch_size > self.node_num else self.batch_size
        self.batch_proc_size = self.batch_size


    def adj_preprocess(self, raw_adj):
        raw_adj_ = raw_adj.detach().clone()
        rowsum = np.array(raw_adj_ + torch.eye(raw_adj_.shape[0])).sum(1)
        degree_mat_inv_sqrt = torch.tensor(np.power(rowsum, -0.5).flatten()).unsqueeze(-1)
        degree_mat_inv_sqrt = degree_mat_inv_sqrt * torch.eye(raw_adj_.shape[0])
        adj_normalized = torch.mm(torch.mm(raw_adj, degree_mat_inv_sqrt).t(), degree_mat_inv_sqrt)
        return adj_normalized

    def discriminate(self, origin_adj, origin_feature, generated_adj, generated_feature):
        origin_adj_norm = self.adj_preprocess(origin_adj)
        origin_embed = self.disc_gcn(origin_feature, origin_adj_norm)
        orig_prob = self.disc_linear(origin_embed)
        generated_adj_norm = self.adj_preprocess(generated_adj)
        generated_embed = self.disc_gcn(generated_feature, generated_adj_norm)
        generated_prob = self.disc_linear(generated_embed)
        return orig_prob, generated_prob


    def encode(self, nodes,for_test):
        _set = set
        _sample = np.random.choice
        node_num = len(self.adj_lists)
        # encode nodes by its neighs
        to_neighs = [self.adj_lists[int(node)] for node in nodes]

        samp_neighs = [(_sample(list(to_neigh), self.enc2.num_sample)) for to_neigh in to_neighs]
        neighs = np.array(samp_neighs).flatten()

        # unique_nodes_list is all nodes required in layer2
        not_unique_nodes_list = list(np.concatenate((neighs, nodes)))

        # encode unique_nodes_list in layer1
        embeds_layer1 = self.enc1(self.features, not_unique_nodes_list, for_test=for_test)

        feature_dict = {}
        subgraph_feature = []
        for i,v in enumerate(not_unique_nodes_list):
            feature_dict[v] = i

        for i,v in enumerate(nodes):
            subgraph_feature.append(self.features_np[v])
        subgraph_feature_tensor = torch.FloatTensor(np.array(subgraph_feature))

        features_embeds = F.relu(self.bn1(embeds_layer1))

        # feed Look-up dict and features_embeds into layer2
        nodes_idx = []
        for i, v in enumerate(nodes):
            nodes_idx.append(feature_dict[v])

        mu = self.enc2(features_embeds, nodes_idx, samp_neighs, feature_dict=feature_dict, for_test=for_test)
        mu = self.bn2(mu)
        logvar = self.enc3(features_embeds, nodes_idx, samp_neighs, feature_dict=feature_dict, for_test=for_test)
        logvar = self.bn3(logvar)

        return mu, logvar, subgraph_feature_tensor


    def decode(self, input, node_list, for_test):
        if not for_test:
            input = input.view(input.shape[0],1,input.shape[1])

        # decoder1
        output = self.dec1(input, for_test)
        output = self.bn4(output)
        output = F.relu(output)

        # decoder2
        output = self.dec2(output, for_test)
        output = self.bn5(output)
        output = F.relu(output)

        # linear transform
        emb1 = self.mapping1(output, for_test)
        emb1 = self.bn6(emb1)
        return emb1


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def inner_product_with_mapping(self, emb1,emb2, nodes):
        adj = torch.mm(emb1, emb2.t())
        adj_without_sigmoid = adj
        return adj_without_sigmoid


    def forward(self, nodes, sub_adj, for_test=False):
        orig_prob, generated_prob = None, None
        mu, logvar, subgraph_feature_tensor = self.encode(nodes, for_test)
        h = self.reparameterize(mu, logvar)
        emb1 = self.decode(h, nodes, for_test)
        adj_without_sigmoid = self.inner_product_with_mapping(emb1, emb1, nodes)
        # the prob of one graph is orig
        if not for_test:
            orig_prob, generated_prob = self.discriminate(sub_adj, subgraph_feature_tensor, adj_without_sigmoid, emb1)
        return mu, logvar, adj_without_sigmoid, orig_prob, generated_prob


    def loss_function(self, preds, labels, mu, logvar, n_nodes, pos_weight, epoch, orig_prob, generated_prob):
        cost1 = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=torch.FloatTensor([pos_weight]) ,size_average=True)
        disc_prob = torch.cat((orig_prob, generated_prob))
        disc_labl = torch.cat((torch.ones_like(orig_prob), torch.zeros_like(generated_prob)))
        cost2 = F.binary_cross_entropy_with_logits(disc_prob, disc_labl)
        return cost1 + self.disc_factor * cost2


        # # see Appendix B from VAE paper:
        # # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # # https://arxiv.org/abs/1312.6114
        # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = -(0.5 / n_nodes) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        # print(KLD)
        # return cost + KLD
