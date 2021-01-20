import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from src.DPGGAN import px_expander
from src.DPGGAN.utils_dp import create_cum_grads, update_privacy_pars, perturb_grad, update_privacy_account
from src.eval import compute_graph_statistics
from src.utils import save_top_n, save_edge_num, sample_subgraph
seed = 3

def generate_graph(adj_without_sigmoid, adj, threshold):
    generated_graph, gene_neck_value = save_top_n(adj_without_sigmoid, save_edge_num(adj), threshold=threshold)
    return generated_graph, gene_neck_value


def eval_generated_graph(args, epoch, model, adj, logger):
    generated_graph_property_cache = []
    params_graph = None
    with torch.no_grad():
        for i in range(args.stat_generate_num):
            # get embedding of all nodes
            node_list = np.array(range(adj.shape[0]))
            # recovered is a sigmoid prob matrix
            mu, logvar, adj_without_sigmoid, _, _ = model.forward(node_list, adj, for_test=True)
            # the generated_graph is a [0,1] matrix. recovered is the mu+var, so re-generate from mu
            params_graph, params_neck_value = generate_graph(adj_without_sigmoid.numpy(), adj, args.threshold)

            generated_graph_property_cache.append(compute_graph_statistics(params_graph))
        stat_log_info = logger.form_generated_stat_log(epoch, generated_graph_property_cache)
        logger.write(stat_log_info)

    return params_graph, generated_graph_property_cache


def get_loss_weight(node_num, dataset):
    pos_weight = float(node_num * node_num - dataset.adj.sum()) / dataset.adj.sum()
    return pos_weight


def train(args, model_args, dataset, model, optimizer, logger):
    node_num, feature_dim = dataset.features.shape
    pos_weight = get_loss_weight(node_num, dataset) # Use global norm and weight

    original_graph_stat = compute_graph_statistics(dataset.adj)
    stat_log_info = logger.form_original_stat_log(original_graph_stat)
    logger.write(stat_log_info)

    stop_info = None
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        index = np.array(list(range(dataset.adj.shape[0])))
        sub_adj = torch.FloatTensor(np.array(dataset.adj.todense()))
        mu, logvar_sub, reconst_adj, gan_pred, gan_label = model.forward(index, sub_adj, for_test=False)
        # Calculate loss
        loss = model.loss_function(reconst_adj, sub_adj, logvar_sub, pos_weight, gan_pred, gan_label)
        # Optimize
        assert torch.isnan(loss).sum() == 0 # if nan appears, change lr
        loss.backward()

        if 'DP' in args.model_name:
            perturb_grad(model_args, model)

        # Perform one step optimize
        if args.optimizer == 'ADADP':
            optimizer.step1() if epoch % 2 == 0 else optimizer.step2(model_args.tol)
        else:
            optimizer.step()

        if 'DP' in args.model_name:
            stop_signal = update_privacy_account(model_args, model) # Update privacy budget
        else:
            stop_signal = False

        if stop_signal:
            stop_info = "Run out of DP budget at epoch@{}.".format(epoch)
            print(stop_info)
            break

    generated_adj, generated_graph_property_cache = eval_generated_graph(args, args.epochs, model, dataset.adj, logger)

    if stop_info is not None:
        logger.write(stop_info)

    logger.write('='*25 + '\n')
    print("Optimization Finished!")
    return model, generated_adj, generated_graph_property_cache



#
# def draw_graph(args, epoch, model, node_num, adj):
#     # No matter whether full_data,
#     # Generate graph from multi-normal distribution and paramterized distribution @epoch
#     if (epoch + 1) % args.draw_graph_epoch == 0 and args.dataset_type == 'single':
#         with torch.no_grad():
#             mu, logvar, adj_without_sigmoid, _ = model.forward(np.array(range(node_num)), args.use_L2_Loss,
#                                                                KL_type=args.KL_type, for_test=True)
#
#             ##### generate from mu and logvar #####
#             params_graph, params_neck_value, _ = \
#                 generate_graph(None, adj_without_sigmoid, model, adj, args, list(range(node_num)),
#                                threshold=args.threshold, for_test=True)
#
#             print('params_graph neck_value@' + str(epoch) + ": " + str(params_neck_value))
#             ######################################
#
#             ##### generate only from mu ########
#             emb_graph, emb_neck_value, _ = \
#                 generate_graph(mu, None, model, adj, args, list(range(node_num)),
#                                threshold=args.threshold, for_test=True)
#
#             print('emb_graph neck_value@' + str(epoch) + ": " + str(emb_neck_value))
#             ###################################·    ###
#
#             ##### generate from random seed ######
#             random_seed = torch.randn(node_num, args.layer2_dim)
#             generated_graph, gene_neck_value, _ = generate_graph(random_seed, None, model, adj, args,
#                                                                  list(range(node_num)),
#                                                                  threshold=args.threshold, for_test=True)
#             print('gene_graph neck_value@' + str(epoch) + ": " + str(gene_neck_value))
#             ######################################
