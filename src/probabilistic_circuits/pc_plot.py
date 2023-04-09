def plot_pc(pc, file_name="pc_plot.pdf", path=None, make_leaves_unique=False):
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from networkx.drawing.nx_pydot import graphviz_layout
    from probabilistic_circuits.pc_nodes import PCSum, PCProduct, PCLeaf
    from util import io

    id_dict = {}
    g = nx.Graph()
    labels = {}

    def _create_graph(node):
        if node not in id_dict:
            id_dict[node] = len(id_dict) + 1
        g.add_node(id_dict[node])
        if isinstance(node, PCSum):
            label = "+"
        elif isinstance(node, PCProduct):
            label = "x"
        else:
            if len(node.scope) == 0:
                label = "1"
            else:
                assert(len(node.scope) == 1)
                label = str(list(node.scope)[0])
        labels[id_dict[node]] = label
        if isinstance(node, PCLeaf):
            return
        for i, child in enumerate(node.children):
            edge_label = ""
            if isinstance(node, PCSum):
                edge_label = np.round(node.weights[i], 3)
            _create_graph(child)
            g.add_edge(id_dict[child], id_dict[node], weight=edge_label)
    _create_graph(pc)

    plt.clf()
    pos = graphviz_layout(g, prog="dot")
    ax = plt.gca()
    nx.draw(g, pos, with_labels=True, arrows=False, node_color="#DDDDDD", edge_color="#888888", width=1, node_size=200, labels=labels, font_size=8)
    ax.collections[0].set_edgecolor("#333333")
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=nx.get_edge_attributes(g, "weight"), font_size=8, alpha=0.6)
    xpos = list(map(lambda p: p[0], pos.values()))
    ypos = list(map(lambda p: p[1], pos.values()))
    ax.set_xlim(min(xpos) - 20, max(xpos) + 20)
    ax.set_ylim(min(ypos) - 20, max(ypos) + 20)
    plt.tight_layout()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    if path is None:
        path = io.get_project_directory() + "_graphs/"
    plt.savefig(path + file_name, bbox_inches="tight", pad_inches=0, dpi=500)
