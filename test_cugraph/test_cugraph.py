# %env NX_CUGRAPH_AUTOCONFIG=True
import networkx as nx
# import cugraph

# 创建一个加权图
G = nx.DiGraph()
G.add_weighted_edges_from([
    (0, 1, 1.0),
    (1, 2, 2.0),
    (2, 3, 3.0),
    
    (3, 4, 4.0),
    (4, 0, 5.0)
])

# 转换为 cuGraph
# G_cu = cugraph.utilities.convert_from_nx(G)

# 尝试计算加权直径
try:
    diameter = nx.diameter(G, weight='weight')
    print("Diameter:", diameter)
except Exception as e:
    print("Error:", e)
