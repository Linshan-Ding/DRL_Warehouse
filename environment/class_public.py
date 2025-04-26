"""
定义了公共的类
"""

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
            "robot": {
                # 机器人单位运行成本
                "unit_run_cost": 1,
                # 机器人移动速度
                "robot_speed": 2
            },
            "picker": {
                # 拣货员单位时间雇佣成本
                "unit_time_cost": 1,
                # 拣货员移动速度
                "picker_speed": 2,
                # 拣货员辞退成本
                "unit_fire_cost": 10
            },
            "order": {
                # 订单单位延期成本
                "unit_delay_cost": 1
            }
        }

        return parameters


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
        # 订单拣选完成时间
        self.complete_time = None

    # 订单延期总成本
    @ property
    def total_delay_cost(self, current_time):
        """
        计算订单延期总成本
        :param current_time: 当前时间
        :return: 订单延期总成本
        """
        if self.complete_time is not None:
            if current_time < self.due_time:
                return 0
            else:
                return (current_time - self.due_time) * self.unit_delay_cost
        else:
            if self.complete_time <= self.due_time:
                return 0
            else:
                return (self.complete_time - self.due_time) * self.unit_delay_cost