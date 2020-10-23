import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from src.DPGraphGen import px_expander
from src.DPGraphGen.utils_dp import create_cum_grads, update_privacy_pars, perturb_grad, update_privacy_account
from src.eval import compute_graph_statistics
from src.utils import save_top_n, save_edge_num, sample_subgraph

def draw_graph(args, epoch, model, node_num, adj):
    # No matter whether full_data,
    # Generate graph from multi-normal distribution and paramterized distribution @epoch
    if (epoch + 1) % args.draw_graph_epoch == 0 and args.dataset_type == 'single':
        with torch.no_grad():
            mu, logvar, adj_without_sigmoid, _ = model.forward(np.array(range(node_num)), args.use_L2_Loss,
                                                               KL_type=args.KL_type, for_test=True)

            ##### generate from mu and logvar #####
            params_graph, params_neck_value, _ = \
                generate_graph(None, adj_without_sigmoid, model, adj, args, list(range(node_num)),
                               threshold=args.threshold, for_test=True)

            print('params_graph neck_value@' + str(epoch) + ": " + str(params_neck_value))
            ######################################

            ##### generate only from mu ########
            emb_graph, emb_neck_value, _ = \
                generate_graph(mu, None, model, adj, args, list(range(node_num)),
                               threshold=args.threshold, for_test=True)

            print('emb_graph neck_value@' + str(epoch) + ": " + str(emb_neck_value))
            ###################################·    ###

            ##### generate from random seed ######
            random_seed = torch.randn(node_num, args.layer2_dim)
            generated_graph, gene_neck_value, _ = generate_graph(random_seed, None, model, adj, args,
                                                                 list(range(node_num)),
                                                                 threshold=args.threshold, for_test=True)
            print('gene_graph neck_value@' + str(epoch) + ": " + str(gene_neck_value))
            ######################################


def generate_graph(seed, adj_without_sigmoid, model, adj, node_list, threshold, for_test=True):
    '''
    Generate graph without create redundant tensor node
    seed: the hidden feature of node, type: tensor
    node_list: the node name, order should be same as seed
    threshold is for sigmoid(adj_with_sigmoid.numpy())
    '''
    if adj_without_sigmoid is None:
        emb1,emb2 = model.decode(seed, node_list, for_test)
        seed_adj = model.inner_product_with_mapping(emb1,emb2, node_list)
    else:
        seed_adj = adj_without_sigmoid

    recovered = torch.sigmoid(seed_adj).data.numpy()
    generated_graph, gene_neck_value = save_top_n(recovered, save_edge_num(adj), threshold=threshold)
    return generated_graph, gene_neck_value, None


def eval_generated_graph(args, epoch, node_num, model, adj, logger, sub_adj):
    # full_data ==>>  Compute graph property for generated graph with N(\mu, \logvar)
    generated_graph_property_cache = []
    with torch.no_grad():
        for i in range(args.stat_generate_num):
            ###########################
            # get embedding of all nodes
            node_list = np.array(range(node_num))

            # recovered is a sigmoid prob matrix
            mu, logvar, adj_without_sigmoid, _, _ = model.forward(node_list, sub_adj, for_test=True)

            # the generated_graph is a [0,1] matrix. recovered is the mu+var, so re-generate from mu
            params_graph, params_neck_value, _ = \
                generate_graph(None, adj_without_sigmoid, model, adj, list(range(node_num)),
                                                                threshold=args.threshold, for_test=True)
            ###########################

            # print('generated_graph neck_value: ' + str(params_neck_value))
            generated_graph_property_cache.append(compute_graph_statistics(params_graph))

        stat_log_info = logger.form_generated_stat_log(epoch, generated_graph_property_cache)
        logger.write(stat_log_info)
    return params_graph


def get_loss_weight(args, node_num, dataset):
    pos_weight = float(node_num * node_num - dataset.adj.sum()) / dataset.adj.sum()
    return pos_weight


def train(args, model_args, dataset, model, optimizer, logger):
    node_num, feature_dim = dataset.features.shape
    pos_weight = get_loss_weight(args, node_num, dataset) # Use global norm and weight

    args_log_info = logger.form_args_log_content(args, model_args)
    logger.write(args_log_info)

    original_graph_stat = compute_graph_statistics(dataset.adj)
    stat_log_info = logger.form_original_stat_log(original_graph_stat)
    logger.write(stat_log_info)

    stop_info = None
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        index, sub_adj = sample_subgraph(args, node_num, dataset)
        sub_adj = torch.FloatTensor(sub_adj)
        mu, logvar, adj_without_sigmoid, orig_prob, generated_prob = model.forward(index, sub_adj, for_test=False)

        # Calculate loss
        loss = model.loss_function(adj_without_sigmoid, sub_adj, mu, logvar,
                                    args.batch_size, pos_weight, epoch, orig_prob, generated_prob)
        # Optimize
        loss.backward()

        if args.model_name == 'DPGraphGen':
            perturb_grad(model_args, model)

        # Perform one step optimize
        if args.optimizer == 'ADADP':
            if epoch % 2 == 0:
                optimizer.step1()
            else:
                optimizer.step2(model_args.tol)
        else:
            optimizer.step()

        stop_signal = update_privacy_account(args, model_args, model) # Update privacy budget

        if stop_signal:
            stop_info = "Run out of DP budget at epoch@{}.".format(epoch)
            print(stop_info)
            break
    generated_adj = eval_generated_graph(args, epoch, node_num, model, dataset.adj, logger, sub_adj)
    if stop_info is not None:
        logger.write(stop_info)

    logger.write('='*25 + '\n')
    print("Optimization Finished!")
    return model, generated_adj