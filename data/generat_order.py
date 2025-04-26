"""
生成仿真订单
"""
from environment.class_public import Order
import random
import copy
import pickle

# 生成数据类
class GenerateData:
    def __init__(self, warehouse, total_seconds, poisson_parameter):
        self.warehouse = warehouse
        self.total_seconds = total_seconds
        self.poisson_parameter = poisson_parameter

    def generate_orders(self):
        """生成订单"""
        # 一个月内的订单列表
        orders = []
        # 订单编号
        order_id = 0
        # 订单到达时间
        arrival_time = 0
        # 仓库中的商品总数
        n_items = len(self.warehouse.items)
        while True:
            # 订单中的商品数量
            order_n_items = random.randint(1, n_items)  # 订单中的商品数量服从均匀分布
            # 仓库中的商品对象列表
            items_list = copy.deepcopy(list(self.warehouse.items.values()))
            # 从仓库所有商品字典中不重复抽样n_items个商品
            items = random.sample(items_list, order_n_items)
            # 深复制items
            items = copy.deepcopy(items)
            # 创建订单对象
            arrival_time += random.expovariate(1 / self.poisson_parameter)  # 订单到达时间服从泊松分布
            # 到达时间取整
            arrival_time = int(arrival_time)  # 订单到达时间
            due_time = arrival_time + random.expovariate(1 / self.poisson_parameter)  # 到期时间
            order_id += 1  # 订单编号
            order = Order(order_id, items, arrival_time, due_time)  # 创建订单对象
            orders.append(order)  # 将订单加入到订单列表中
            # 若订单到达时间大于一个月的总秒数，则跳出循环
            if arrival_time >= self.total_seconds:
                break

        # 将orders信息保存到D:\Python project\DRL_Warehouse\data文件夹中，并在命名中融合self.poisson_parameter信息
        with open("D:\Python project\DRL_Warehouse\data\orders_{}.pkl".format(self.poisson_parameter), "wb") as f:
            pickle.dump(orders, f)

        print(f"Total number of orders: {len(orders)}")