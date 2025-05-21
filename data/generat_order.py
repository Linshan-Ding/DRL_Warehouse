"""
生成仿真订单
"""
from environment.class_public import Order
import random
import copy
import pickle
from data.warehouse import WarehouseEnv
from environment.class_public import Config

# 生成数据类
class GenerateData(Config):
    def __init__(self, warehouse, total_seconds, parameter=None):
        super().__init__()  # 调用父类的构造函数
        self.parameter = self.parameters["order"]  # 订单参数
        self.warehouse = warehouse  # 仓库对象
        self.total_seconds = total_seconds  # 一个月的总秒数
        if parameter is None:
            # 默认泊松分布参数
            self.poisson_parameter = random.randint(self.parameter["poisson_parameter"][0], self.parameter["poisson_parameter"][1])  # 泊松分布参数, n秒一个订单到达
            self.save_data = False
        else:
            self.poisson_parameter = parameter  # 泊松分布参数, n秒一个订单到达
            self.save_data = True

    def generate_orders(self):
        """生成订单"""
        # 一个月内的订单列表
        orders = []
        # 订单编号
        order_id = 0
        # 订单到达时间
        arrival_time = 0
        while True:
            # 订单中的商品数量
            order_n_items = random.randint(self.parameter["order_n_items"][0], self.parameter["order_n_items"][1])
            # 仓库中的商品对象列表
            items_list = copy.deepcopy(list(self.warehouse.items.values()))
            # 从仓库所有商品字典中不重复抽样n_items个商品
            items = random.sample(items_list, order_n_items)
            # 深复制items
            items = copy.deepcopy(items)
            # 创建订单对象
            arrival_time += random.expovariate(1 / self.poisson_parameter)  # 订单到达时间服从泊松分布
            due_time = arrival_time + random.choice(self.parameter["due_time_list"])  # 订单交期
            order_id += 1  # 订单编号
            order = Order(order_id, items, arrival_time, due_time)  # 创建订单对象
            orders.append(order)  # 将订单加入到订单列表中
            # 若订单到达时间大于一个月的总秒数，则跳出循环
            if arrival_time >= self.total_seconds:
                break

        if self.save_data:
            # 将orders信息保存到D:\Python project\DRL_Warehouse\data文件夹中，并在命名中融合self.poisson_parameter信息
            with open("D:\Python project\DRL_Warehouse\data\orders_{}.pkl".format(self.poisson_parameter), "wb") as f:
                pickle.dump(orders, f)
            print(f"Total number of orders: {len(orders)}")

        return orders


if __name__ == "__main__":
    # 实例化仓库对象
    warehouse = WarehouseEnv()
    # 一个月的总秒数
    total_seconds = 31 * 8 * 3600  # 31天

    # 订单数据保存和读取位置
    file_order = 'D:\Python project\DRL_Warehouse\data'
    # 订单到达泊松分布参数
    poisson_parameter = 30  # 泊松分布参数, n秒一个订单到达

    # 生成一个月内的订单数据，并保存到orders.pkl文件中
    generate_orders = GenerateData(warehouse, total_seconds, poisson_parameter)  # 生成订单数据对象
    orders = generate_orders.generate_orders()  # 生成一个月内的订单数据

    # # 读取一个月内的订单数据，orders.pkl文件中
    # with open(file_order + "\orders_{}.pkl".format(poisson_parameter), "rb") as f:
    #     orders = pickle.load(f)  # 读取订单数据