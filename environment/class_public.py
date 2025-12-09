"""
参数定义类
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
                # 单层货架中储货位数量
                "shelf_capacity": 20,
                # 货架层数
                "shelf_levels": 3,
                # 仓库区域数量
                "area_num": 3,
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
                "depot_position": (18, 0)
            },
            "robot": {
                # 短租机器人单位运行成本 0.0038
                "short_term_unit_run_cost": 110/(3600*8),
                # 长租机器人单位运行成本 0.00039
                "long_term_unit_run_cost": 1000000/(3600*8*30*8*365),
                # 机器人移动速度 m/s
                "robot_speed": 1.2
            },
            "picker": {
                # 短租拣货员单位时间雇佣成本 0.0125元/秒
                "short_term_unit_time_cost": 360/(3600*8),
                # 长租拣货员单位时间雇佣成本 0.0081元/秒
                "long_term_unit_time_cost": 7000/(3600*8*30),
                # 拣货员移动速度 m/s
                "picker_speed": 0.75,
                # 拣货员辞退成本 元
                "unit_fire_cost": 0
            },
            "order": {
                # 订单单位延期成本 元/秒
                "unit_delay_cost": 0.05,  # 元/秒
                # 订单打包时间 秒
                "pack_time": 20,  # 秒
                # 订单到达率范围 秒/个 相当于泊松分布参数
                "poisson_parameter": (60, 180),  # 秒/个
                # 订单从到达到交期的可选时间长度列表 秒
                "due_time_list": [1800, 3600, 7200],  # 秒
                # 每次到达的订单数量范围 个
                "order_n_arrival": (1, 10),  # 个
                # 单个订单包含的商品数量范围 个
                "order_n_items": (10, 30)  # 个
            },
            "item": {
                # 商品拣选时间
                "pick_time": 10  # 秒
            },
            "ppo": {
                # PPO算法参数
                "gamma": 1,  # 折扣因子
                "clip_range": 0.2,  # 剪切范围
                "learning_rate": 3e-4,  # 学习率
                "n_epochs": 3,  # 每个批次的训练轮数
                "normalize_rewards": True,  # 是否归一化回报
                "standardize_rewards": True,  # 是否标准化回报
                "initial_entropy_coeff": 0.05,  # 初始熵系数
                "min_entropy_coeff": 0.001,  # 最小熵系数
                "entropy_coeff_decay": 0.995  # 熵衰减率
            }
        }

        return parameters