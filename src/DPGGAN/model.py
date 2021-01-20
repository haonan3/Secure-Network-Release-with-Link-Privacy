import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from src.DPGGAN.DPCounter import DPCounter
from src.DPGGAN.dp_aggregators import MeanAggregator as DPMeanAggregator
from src.DPGGAN.dp_encoders import Encoder as DPEncoder
from src.DPGGAN.gcn_layer import GraphConvolution
# from src.GGAN.aggregators import MeanAggregator
# from src.GGAN.encoders import Encoder
import src.DPGGAN.linear as linear
from src.DPGGAN.utils_dp import create_cum_grads


class DPGGAN(nn.Module):
    def __init__(self, args, model_args, features_np, adj_lists):
        super(DPGGAN, self).__init__()
        self.args = args
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


        # Follow the mode used in graphSAGE, the agg with 'gcn=False'
        # layer1
        self.agg1 = DPMeanAggregator(cuda=False,first_layer=True,gcn=False)
        self.enc1 = DPEncoder(self.feature_dim, self.layer1_dim,
                              self.adj_lists, self.agg1, num_sample=self.samp_num,
                            gcn=True, cuda=False, first_layer=True, batch_size=self.batch_size*(self.samp_num+1))

        # layer2
        self.agg2 = DPMeanAggregator(cuda=False,first_layer=False,gcn=False)
        self.enc2 = DPEncoder(self.layer1_dim, self.layer2_dim,
                              self.adj_lists, self.agg2, num_sample=self.samp_num,
                            gcn=True, cuda=False, first_layer=False, batch_size=self.batch_size)

        self.agg3 = DPMeanAggregator(cuda=False,first_layer=False,gcn=False)
        self.enc3 = DPEncoder(self.layer1_dim, self.layer2_dim,
                              self.adj_lists, self.agg3, num_sample=self.samp_num,
                            gcn=True, cuda=False, first_layer=False, batch_size=self.batch_size)

        # decoder layer1
        self.dec1 = linear.Linear(self.layer2_dim, self.dec1_dim, bias=False, batch_size=self.batch_size)
        # decoder layer2
        self.dec2 = linear.Linear(self.dec1_dim, self.dec2_dim, bias=False, batch_size=self.batch_size)

        self.mapping1 = linear.Linear(self.dec2_dim, self.dec2_dim, bias=False, batch_size=self.batch_size)
        self.mapping2 = linear.Linear(self.dec2_dim, self.dec2_dim, bias=False, batch_size=self.batch_size)

        self.cum_grads = create_cum_grads(self)

        # gan part
        self.is_gan = 0
        if args.model_name == 'DPGGAN':
            self.disc_gcn = GraphConvolution(in_features=self.dec2_dim, out_features=self.dec2_dim)
            self.disc_linear = nn.Linear(in_features=self.dec2_dim, out_features=1, bias=False)
            self.is_gan = 1

    def check_proc_size(self):
        self.batch_size = self.node_num if self.batch_size > self.node_num else self.batch_size
        self.batch_proc_size = self.batch_size



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
        for i,v in enumerate(not_unique_nodes_list):
            feature_dict[v] = i

        features_embeds = F.relu(embeds_layer1)

        # feed Look-up dict and features_embeds into layer2
        nodes_idx = []
        for i, v in enumerate(nodes):
            nodes_idx.append(feature_dict[v])

        mu = self.enc2(features_embeds, nodes_idx, samp_neighs, feature_dict=feature_dict, for_test=for_test)
        logvar = self.enc3(features_embeds, nodes_idx, samp_neighs, feature_dict=feature_dict, for_test=for_test)

        return mu, logvar


    def decode(self, input, for_test):
        if not for_test:
            input = input.view(input.shape[0],1,input.shape[1])

        # decoder1
        output = self.dec1(input, for_test)
        output = F.normalize(output, dim=-1, p=2)
        output = F.relu(output)

        # decoder2
        output = self.dec2(output, for_test)
        output = F.normalize(output, dim=-1, p=2)
        output = F.relu(output)

        # linear transform
        emb1 = self.mapping1(output, for_test)
        emb2 = self.mapping2(output, for_test)

        return emb1, emb2


    def inner_product_with_mapping(self, emb1, emb2):
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
        mu, logvar = self.encode(nodes, for_test)

        mu_q = F.normalize(mu, dim=-1, p=2)
        logvar_sub = -logvar
        std0 = self.args.std
        std_q = torch.exp(0.5 * logvar_sub) * std0
        epsilon = torch.randn(std_q.shape)
        h = mu_q + int(for_test) * epsilon * std_q
        emb1, emb2 = self.decode(h, for_test)

        reconst_adj = self.inner_product_with_mapping(emb1, emb2)
        # the prob of one graph is orig
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