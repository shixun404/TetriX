import networkx as nx

def compare_graph_edges(sG, G, weight_key='weight'):
    """
    比较两个 nx 图 sG 和 G 的边及其权重，并打印差异。

    参数:
    - sG, G: 两个待比较的 networkx 图 (DiGraph 或 Graph 均可)
    - weight_key: 边权对应的属性名(默认为 'weight')
    """
    # 将所有边存进字典, 便于查找
    # 格式: edges_dict_sG[(u,v)] = weight
    edges_dict_sG = {}
    for u, v, data in sG.edges(data=True):
        w = data.get(weight_key, None)  # 无weight时返回None
        edges_dict_sG[(u, v)] = w

    edges_dict_G = {}
    for u, v, data in G.edges(data=True):
        w = data.get(weight_key, None)
        edges_dict_G[(u, v)] = w

    # 1) 检查 sG 中的边是否存在于 G
    for (u, v), w_sG in edges_dict_sG.items():
        if (u, v) not in edges_dict_G:
            print(f"Edge ({u}, {v}) in sG but not in G. (weight={w_sG})")
        else:
            w_G = edges_dict_G[(u, v)]
            if w_sG != w_G:
                print(f"Edge ({u}, {v}) weight different: sG={w_sG}, G={w_G}")

    # 2) 检查 G 中的边是否存在于 sG
    for (u, v), w_G in edges_dict_G.items():
        if (u, v) not in edges_dict_sG:
            print(f"Edge ({u}, {v}) in G but not in sG. (weight={w_G})")