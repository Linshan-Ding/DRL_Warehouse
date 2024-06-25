import simpy
import random
import numpy as np
import networkx as nx
from scipy.stats import poisson

# 仓库参数
N_l = 10  # Rack中的储货位数量
N_w = 5   # Rack数量
S_l = 1   # 储货位长
S_w = 1   # 储货位宽
S_b = 2   # 底部横向巷道宽
S_d = 2   # 最左侧纵向巷道宽
S_a = 2   # 纵向巷道宽
L = S_b + N_l * S_l  # 系统长度
W = S_d + N_w * (S_w + S_a)  # 系统宽度
depot_position = ((W - S_d) / 2, 0)

# 创建仓库图
def create_warehouse_graph(N_l, N_w, S_l, S_w, S_b, S_d, S_a):
    G = nx.Graph()
    # Add nodes for each pick position
    for nw in range(1, N_w + 1):
        for nl in range(1, N_l + 1):
            x = S_d + (2 * nw - 1) * S_w + (2 * nw - 1) / 2 * S_a
            y = S_b + (2 * nl - 1) / 2 * S_l
            G.add_node((x, y))
    # Add edges for horizontal movement
    for nl in range(1, N_l + 1):
        for nw in range(1, N_w):
            x1 = S_d + (2 * nw - 1) * S_w + (2 * nw - 1) / 2 * S_a
            y1 = S_b + (2 * nl - 1) / 2 * S_l
            x2 = S_d + (2 * (nw + 1) - 1) * S_w + (2 * (nw + 1) - 1) / 2 * S_a
            G.add_edge((x1, y1), (x2, y1))
    # Add edges for vertical movement
    for nw in range(1, N_w + 1):
        for nl in range(1, N_l):
            x = S_d + (2 * nw - 1) * S_w + (2 * nw - 1) / 2 * S_a
            y1 = S_b + (2 * nl - 1) / 2 * S_l
            y2 = S_b + (2 * (nl + 1) - 1) / 2 * S_l
            G.add_edge((x, y1), (x, y2))
    return G

warehouse_graph = create_warehouse_graph(N_l, N_w, S_l, S_w, S_b, S_d, S_a)

class WarehouseEnv:
    def __init__(self, env, num_robots, order_rate):
        self.env = env
        self.num_robots = num_robots
        self.order_rate = order_rate
        self.robots = [self.Robot(env, i, warehouse_graph, depot_position) for i in range(num_robots)]
        self.orders = simpy.Store(env)
        self.env.process(self.order_arrival_process())
        for robot in self.robots:
            self.env.process(robot.run())

    class Robot:
        def __init__(self, env, robot_id, graph, depot_position):
            self.env = env
            self.robot_id = robot_id
            self.graph = graph
            self.depot_position = depot_position
            self.position = depot_position
            self.order = None

        def run(self):
            while True:
                # Wait for an order
                self.order = yield self.env.process(self.get_order())
                print(f"Robot {self.robot_id} received order {self.order} at time {self.env.now}")

                # Process the order
                for item in self.order['items']:
                    pick_position = item['pick_position']
                    path = nx.shortest_path(self.graph, source=self.position, target=pick_position)
                    for step in path:
                        self.position = step
                        yield self.env.timeout(1)  # Simulate movement
                        print(f"Robot {self.robot_id} moved to {self.position} at time {self.env.now}")
                    yield self.env.timeout(2)  # Simulate picking time
                    print(f"Robot {self.robot_id} picked item at {self.position} at time {self.env.now}")

                # Return to depot
                path = nx.shortest_path(self.graph, source=self.position, target=self.depot_position)
                for step in path:
                    self.position = step
                    yield self.env.timeout(1)
                    print(f"Robot {self.robot_id} returned to {self.position} at time {self.env.now}")

                # Ready for the next order
                self.order = None

        def get_order(self):
            return self.env.process(simulation.orders.get())

    def order_arrival_process(self):
        while True:
            yield self.env.timeout(random.expovariate(self.order_rate))
            order_id = self.env.now
            num_items = random.randint(1, 5)
            items = [{'pick_position': random.choice(list(self.robots[0].graph.nodes))} for _ in range(num_items)]
            order = {'order_id': order_id, 'items': items}
            print(f"Order {order_id} arrived at time {self.env.now}")
            self.orders.put(order)

# 仿真参数
num_robots = 3
order_rate = 0.1  # λ = 0.1

# 仿真环境
env = simpy.Environment()
simulation = WarehouseEnv(env, num_robots, order_rate)
env.run(until=100)
