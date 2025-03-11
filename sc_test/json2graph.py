import json
import networkx as nx
import pickle as pkl
import itertools

# **读取 JSON 数据**
with open("FABRIC_100.json", "r") as f:  # 你的 JSON 文件
    lines = f.read()

# **解析 JSON**
try:
    data = json.loads(lines)
except json.JSONDecodeError as e:
    print("JSON 解析失败:", e)
    exit(1)  # 退出程序

# **提取所有 IP:PORT 作为节点**
def extract_node_id(node_key):
    return node_key.strip()  # 直接使用 "IP:PORT" 作为节点名

# **获取所有唯一节点**
all_nodes = {extract_node_id(node) for node in data.keys()}  # 提取所有 IP:PORT
# 400
all_nodes.add("10.131.130.2:1234")
all_nodes.add("10.141.3.2:1238")
# all_nodes.add("10.140.3.2:1248")

# 400
# all_nodes.add("10.131.130.2:1234")
# all_nodes.add("10.145.2.2:1241")
# all_nodes.add("10.140.3.2:1248")
# **创建 IP:PORT → 自然数 映射**
node_mapping = {node: idx for idx, node in enumerate(sorted(all_nodes))}  # 自然数从 0 开始

# **初始化无向完全图 (Complete Graph)，所有边的 weight = 1000**
G = nx.Graph()
for u, v in itertools.combinations(all_nodes, 2):  # 组合所有可能的 (u, v) 边
    G.add_edge(node_mapping[u], node_mapping[v], weight=1000)

# **更新 JSON 提供的边权重**
for src, neighbors in data.items():
    src_node = node_mapping[extract_node_id(src)]  # 源节点转换为自然数

    for dst, weight in neighbors.items():
        dst_node = node_mapping[extract_node_id(dst)]  # 目标节点转换为自然数
        if G.has_edge(src_node, dst_node):  # 如果边存在，则更新权重
            G[src_node][dst_node]['weight'] = float(weight)

# **保存加权图**
with open("G_100_FABRIC.pkl", "wb") as f:
    pkl.dump(G, f)

# **保存节点映射 (IP:PORT → 自然数)**
with open("node_mapping_100.pkl", "wb") as f:
    pkl.dump(node_mapping, f)

# **输出图信息**
# print(nx.info(G))

# **示例：打印前 10 条边**
for edge in list(G.edges(data=True))[:10]:
    print(edge)

# **示例：打印前 10 个 IP:PORT 对应的编号**
print("\nNode Mapping (IP:PORT -> Natural Number):")
for ip_port, idx in list(node_mapping.items())[:10]:
    print(f"{ip_port} -> {idx}")
