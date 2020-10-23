class drawer:
    def __init__(self):
        pass

    def check_image_save_dir(self):
        pass
        # check the image save dir, according to the timestamp

    def save_graph(self):
        pass
        # save graph to relevant dir
        # write according to the following function.



def draw_graph(adj, path, plot_name, circle, color):
    G = None
    if scipy.sparse.issparse(adj):
        G = nx.from_scipy_sparse_matrix(adj)
    else:
        G = nx.from_numpy_matrix(adj)

    options = {
        'node_color': 'black',
        'node_size': 5,
        'line_color': 'grey',
        'linewidths': 0.1,
        'width': 0.1,
    }

    if circle:
        node_list = sorted(G.degree, key=lambda x: x[1], reverse=True)
        node2order = {}
        for i, v in enumerate(node_list):
            node2order[v[0]] = i

        new_edge = []
        for i in G.edges():
            new_edge.append((node2order[i[0]], node2order[i[1]]))

        new_G = nx.Graph()
        new_G.add_nodes_from(range(len(node_list)))
        new_G.add_edges_from(new_edge)

        nx.draw_circular(new_G, with_labels=True)

    elif color is not None:
        nx.draw(G, node_color=color, node_size=20,  with_labels=True)
    else:
        nx.draw(G, **options)
    plt.savefig(path + '/' + plot_name + ".png")
    plt.clf()