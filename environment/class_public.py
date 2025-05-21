"""
定义了公共的类
"""
# -------------参数定义类----------------
class Config:
    def __init__(self):
        """
        配置类
        """
        self.parameters = self.parameter()  # 配置项

    def parameter(self):
        """
        算法和环境参数
        """
        parameters = {
            "warehouse": {
                # 单个货架中储货位数量
                "shelf_capacity": 30,
                # 仓库区域数量
                "area_num": 6,
                # 仓库每个区域中巷道数量
                "aisle_num": 3,
                # 储货位的长度
                "shelf_length": 1,
                # 储货位的宽度
                "shelf_width": 1,
                # 底部通道的宽度
                "aisle_width": 2,
                # 仓库的出入口处的宽度
                "entrance_width": 2,
                # 巷道的宽度
                "aisle_width": 2,
                # depot_position: 机器人的起始位置
                "depot_position": (0, 0)
            },
            "robot": {
                # 短租机器人单位运行成本
                "short_term_unit_run_cost": 110/(3600*8),
                # 长租机器人单位运行成本
                "long_term_unit_run_cost": 1000000/(3600*8*30*8*365),
                # 机器人移动速度 m/s
                "robot_speed": 1.5
            },
            "picker": {
                # 短租拣货员单位时间雇佣成本 元/秒
                "short_term_unit_time_cost": 360/(3600*8),
                # 长租拣货员单位时间雇佣成本 元/秒
                "long_term_unit_time_cost": 7000/(3600*8*30),
                # 拣货员移动速度 m/s
                "picker_speed": 0.67,
                # 拣货员辞退成本 元
                "unit_fire_cost": 0
            },
            "order": {
                # 订单单位延期成本 元/秒
                "unit_delay_cost": 0.1,  # 元/秒
                # 订单打包时间 秒
                "pack_time": 20,  # 秒
                # 订单到达率范围 秒/个 相当于泊松分布参数
                "poisson_parameter": (6, 30),  # 秒/个
                # 订单从到达到交期的可选时间长度列表 秒
                "due_time_list": [1800, 3600, 7200],  # 秒
                # 每次到达的订单数量范围 个
                "order_n_arrival": (1, 10),  # 个
                # 单个订单包含的商品数量范围 个
                "order_n_items": (10, 20)  # 个
            },
            "item": {
                # 商品拣选时间
                "pick_time": 10  # 秒
            },
            "ppo": {
                # PPO算法参数
                "gamma": 0.99,  # 折扣因子
                "clip_range": 0.2,  # 剪切范围
                "learning_rate": 3e-4,  # 学习率
                "n_epochs": 10,  # 每个批次的训练轮数
            }
        }

        return parameters

# -------------订单类----------------
class Order(Config):
    def __init__(self, order_id, items, arrive_time=0, due_time=None):
        """
        订单类
        :param order_id:  订单编号
        :param items:  订单中的商品列表
        :param arrive_time:  订单到达时间
        :param due_time:  订单交期
        """
        super().__init__()  # 调用父类的构造函数
        self.parameter = self.parameters["order"]  # 订单参数
        self.order_id = order_id  # 订单的编号
        self.items = items  # 订单中的商品列表
        self.arrive_time = arrive_time  # 订单到达时间
        self.due_time = None  # 订单交期
        self.complete_time = None  # 订单拣选完成时间
        # 订单中的未拣选完成的商品列表
        self.unpicked_items = items
        # 订单中的已拣选完成的商品列表
        self.picked_items = []
        # 订单交期
        self.due_time = due_time
        # 订单单位延期成本
        self.unit_delay_cost = self.parameter["unit_delay_cost"]

    # 订单延期总成本
    def total_delay_cost(self, current_time):
        """
        计算订单延期总成本
        :param current_time: 当前时间
        :return: 订单延期总成本
        """
        if self.complete_time is None:
            if current_time < self.due_time:
                return 0
            else:
                return (current_time - self.due_time) * self.unit_delay_cost
        else:
            if self.complete_time <= self.due_time:
                return 0
            else:
                return (self.complete_time - self.due_time) * self.unit_delay_cost