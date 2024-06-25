import simpy
import random
import networkx as nx
import matplotlib.pyplot as plt


def create_warehouse_graph(N_l, N_w, S_l, S_w, S_b, S_d, S_a):
    G = nx.Graph()
    for nw in range(1, N_w + 1):
        for nl in range(1, N_l + 1):
            x = S_d + (2 * nw - 1) * S_w + (2 * nw - 1) / 2 * S_a
            y = S_b + (2 * nl - 1) / 2 * S_l
            G.add_node((x, y))
    for nl in range(1, N_l + 1):
        for nw in range(1, N_w):
            x1 = S_d + (2 * nw - 1) * S_w + (2 * nw - 1) / 2 * S_a
            y1 = S_b + (2 * nl - 1) / 2 * S_l
            x2 = S_d + (2 * (nw + 1) - 1) * S_w + (2 * (nw + 1) - 1) / 2 * S_a
            G.add_edge((x1, y1), (x2, y1))
    for nw in range(1, N_w + 1):
        for nl in range(1, N_l):
            x = S_d + (2 * nw - 1) * S_w + (2 * nw - 1) / 2 * S_a
            y1 = S_b + (2 * nl - 1) / 2 * S_l
            y2 = S_b + (2 * (nl + 1) - 1) / 2 * S_l
            G.add_edge((x, y1), (x, y2))
    return G

# 仓库参数
N_l = 10  # 单个货架中储货位的数量，这里假设是10个储货位
N_w = 6  # 巷道的数量，这里假设是5个巷道，每个巷道两侧各有一个货架，共12个货架
S_l = 1  # 储货位的长度
S_w = 1  # 储货位的宽度
S_b = 2  # 底部通道的宽度
S_d = 2  # 仓库的出入口处的宽度
S_a = 2  # 巷道的宽度
warehouse_graph = create_warehouse_graph(N_l, N_w, S_l, S_w, S_b, S_d, S_a)

# 可视化仓库图
pos = {node: node for node in warehouse_graph.nodes()}
nx.draw(warehouse_graph, pos, with_labels=True, node_size=300, node_color="skyblue")
plt.show()
