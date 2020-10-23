def test(args, current_time, visual_path, result_filename, model, data_package,degree_distribution_folder=None):

    # unpack data_package
    adj_train = data_package['adj_train']
    adj_orig = data_package['adj_orig']
    test_edges = data_package['test_edges']
    test_edges_false = data_package['test_edges_false']
    color = data_package['color']
    node_num = data_package['node_num']
    priv_pars = data_package['priv_pars']

    with torch.no_grad():
        node_list = np.array(range(node_num))
        mu, logvar, adj_without_sigmoid, _ = model.forward(node_list, use_L2_Loss=args.use_L2_Loss, KL_type=args.KL_type, for_test=True)

        # not full_data ==>> link prediction test
        if not args.full_data and args.single_graph:
            roc_score, ap_score = get_roc_score(adj_without_sigmoid, adj_orig, test_edges, test_edges_false)

            print('Test ROC score: ' + str(roc_score))
            print('Test AP score: ' + str(ap_score))
            log_info = 'Test ROC score: ' + str(roc_score) + '\tTest AP score: ' + str(ap_score)
            log_write(result_filename, log_info)


        # full_data ==>> generate some graph after training, in case of sampling variance.
        elif args.single_graph:
            for i in range(args.generated_graph_num):

                ## 1.generated graph ##
                seed = torch.randn(node_num, args.layer2_dim)
                generated_graph, gene_neck_value, _ = generate_graph(seed, None, model, adj_train, args,
                                                                     list(range(node_num)), threshold=args.threshold,
                                                                     for_test=True)

                generated_graph_property = compute_graph_statistics(generated_graph)
                print(generated_graph_property)

                graph_title = args.model_name+"_"+str(args.dataset_str)+'_predict_graph_test_'+str(i)
                draw_graph(generated_graph, visual_path, graph_title, circle=args.circle_plot, color=color)

                if args.single_graph:
                    draw_degree_distribution(generated_graph, args.model_name+'_predict_graph_test_'+str(i),
                                                                                degree_distribution_folder)
                ####################

                ## 2.params_graph ##
                params_graph, params_graph_neck_value, _ = generate_graph(None, adj_without_sigmoid, model, adj_train,
                                                                             args,
                                                                             list(range(node_num)),
                                                                             threshold=args.threshold, for_test=True)

                params_graph_property = compute_graph_statistics(params_graph)
                print(params_graph_property)

                graph_title = args.model_name + "_" + str(args.dataset_str) + '_params_graph_test_' + str(i)
                draw_graph(params_graph, visual_path, graph_title, circle=args.circle_plot, color=color)

                if args.single_graph:
                    draw_degree_distribution(params_graph, args.model_name+'_params_graph_test_ ' + str(i),
                                                                                    degree_distribution_folder)
                ####################

            ## 3.embed_graph ##
            embed_graph_test, embed_test_neck_value, _ = generate_graph(mu, None, model, adj_train, args,
                                                              list(range(node_num)), threshold=args.threshold, for_test=True)

            embed_graph_test_property = compute_graph_statistics(embed_graph_test)
            print(embed_graph_test_property)

            graph_title = args.model_name + "_" + str(args.dataset_str) + '_embed_graph_test'
            draw_graph(embed_graph_test, visual_path, graph_title, circle=args.circle_plot, color=color)

            if args.single_graph:
                draw_degree_distribution(embed_graph_test, args.model_name + '_embed_graph_test',
                                         degree_distribution_folder)
            ####################


            # save 'embed_graph_test' with networkx for presentation
            G = nx.from_numpy_array(embed_graph_test)
            file = sys.path[1] + '/presentation/'
            file = file + args.dataset_str + '_' + args.model_name + '_' + current_time +  '_test.pickle'
            with open(file, 'wb') as handle:
                pickle.dump(G, handle)