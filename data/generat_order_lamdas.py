"""
生成仿真订单
"""
import random
import copy
import pickle
from data.warehouse import WarehouseEnv
from environment.class_public import Config
from environment.class_warehouse import Order

# 生成数据类
class GenerateData(Config):
    def __init__(self, warehouse, total_seconds, parameter_list=None, order_n_items=None):
        super().__init__()  # 调用父类的构造函数
        self.parameter = self.parameters["order"]  # 订单参数
        self.warehouse = warehouse  # 仓库对象
        self.total_seconds = total_seconds  # 一个月的总秒数
        self.poisson_parameters = parameter_list  # 泊松分布参数, n秒一个订单到达
        self.save_data = True
        self.order_n_items = order_n_items  # 订单中的商品数量
        self.day_time = 8 * 3600 # 每天的时常

    def generate_orders(self):
        """生成订单"""
        # 一个月内的订单列表
        orders = []
        # 订单编号
        order_id = 0
        # 订单到达时间
        arrival_time = 0
        while True:
            if self.order_n_items is None:
                self.order_n_items = random.randint(self.parameter["order_n_items"][0], self.parameter["order_n_items"][1])
            # 仓库中的商品对象列表
            items_list = copy.deepcopy(list(self.warehouse.items.values()))
            # 从仓库所有商品字典中不重复抽样n_items个商品
            n_items = random.choice([self.order_n_items - 1, self.order_n_items, self.order_n_items + 1])
            items = random.sample(items_list, n_items)
            # 深复制items
            items = copy.deepcopy(items)
            # 计算订单到达时刻为第几天
            day_count = int(arrival_time // self.day_time)
            # 创建订单对象
            arrival_time += random.expovariate(1 / self.poisson_parameters[day_count])  # 订单到达时间服从泊松分布
            due_time = arrival_time + random.choice(self.parameter["due_time_list"])  # 订单交期
            order_id += 1  # 订单编号
            order = Order(order_id, items, arrival_time, due_time)  # 创建订单对象
            orders.append(order)  # 将订单加入到订单列表中
            # 若订单到达时间大于一个月的总秒数，则跳出循环
            if arrival_time >= self.total_seconds:
                break

        if self.save_data:
            # 将orders信息保存到D:\Python project\DRL_Warehouse\data文件夹中，并在命名中融合self.poisson_parameter信息
            with open("D:\\Python project\\DRL_Warehouse\\data\\instances\\orders_{}.pkl".format(self.order_n_items), "wb") as f:
                pickle.dump(orders, f)
            print(f"Total number of orders: {len(orders)}")

        return orders


if __name__ == "__main__":
    # 实例化仓库对象
    warehouse = WarehouseEnv()
    print('仓库中的商品种类数:', len(warehouse.items))  # 仓库中的商品种类数
    # 一个月的总秒数
    total_seconds = (8 * 3600) * 30  # 31天
    # 订单数据保存和读取位置
    file_order = 'D:\\Python project\\DRL_Warehouse\\data'

    poisson_parameter_list = [121.07, 106.80, 111.69, 120.85, 115.52, 148.70, 143.27, 117.42, 123.36, 117.72,
                              114.31, 129.79, 117.99, 117.14, 105.19, 84.40, 100.79, 121.58, 200.07, 247.73,
                              237.06, 236.22, 154.48, 152.36, 149.37, 167.13, 134.47, 146.41, 134.98, 115.45, 115.45] # 泊松分布参数列表
    order_n_items_list = [10] # 订单中的商品数量列表

    for order_n_items in order_n_items_list:
        # 生成一个月内的订单数据，并保存到orders.pkl文件中
        generate_orders = GenerateData(warehouse, total_seconds, poisson_parameter_list, order_n_items)
        orders = generate_orders.generate_orders()  # 生成持续时间内的订单数据

    # # 读取一个月内的订单数据，orders.pkl文件中
    # with open(file_order + "\orders_{}.pkl".format(poisson_parameter), "rb") as f:
    #     orders = pickle.load(f)  # 读取订单数据