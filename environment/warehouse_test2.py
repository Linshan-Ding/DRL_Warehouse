"""
智能仓库人机协同拣选系统仿真环境
1、待分配拣货位选择最近的空闲拣货员
2、机器人和拣货员被移除后，完成当前订单拣选任务后，再移除
3、每个区最少一个拣货员，整个仓库最少一个机器人
4、同一拣货位的不同商品的拣选时间需要叠加
5、机器人移动到depot_position后，进行打包操作，打包时间为定值
"""
import numpy as np
import random
import pickle
import gymnasium as gym
import copy


# ================= 1. 配置参数类 =================
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
                # 短租机器人单位运行成本
                "short_term_unit_run_cost": 110 / (3600 * 8),
                # 长租机器人单位运行成本
                "long_term_unit_run_cost": 1000000 / (3600 * 8 * 30 * 8 * 365),
                # 机器人移动速度 m/s
                "robot_speed": 1.2
            },
            "picker": {
                # 短租拣货员单位时间雇佣成本 0.0125元/秒
                "short_term_unit_time_cost": 360 / (3600 * 8),
                # 长租拣货员单位时间雇佣成本 0.0081元/秒
                "long_term_unit_time_cost": 7000 / (3600 * 8 * 30),
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


# ================= 2. 实体类定义 =================

# 订单类
class Order(Config):
    def __init__(self, order_id, items, arrive_time=0, due_time=None):
        super().__init__()
        self.parameter = self.parameters["order"]
        self.order_id = order_id
        self.items = items
        self.arrive_time = arrive_time
        self.due_time = due_time
        self.complete_time = None
        self.unpicked_items = items
        self.picked_items = []
        self.unit_delay_cost = self.parameter["unit_delay_cost"]

    def total_delay_cost(self, current_time):
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


# 商品类
class Item(Config):
    def __init__(self, item_id, bin_id, position, area_id, pick_point_id):
        super().__init__()
        self.parameter = self.parameters["item"]
        self.item_id = item_id
        self.bin_id = bin_id
        self.position = position
        self.area_id = area_id
        self.pick_point_id = pick_point_id
        self.pick_time = self.parameter["pick_time"]
        self.pick_complete_time = 0


# 起始点类
class Depot:
    def __init__(self, position):
        self.position = position


# 储货位类
class StorageBin:
    def __init__(self, bin_id, position, area_id, item_id, pick_point_id):
        self.bin_id = bin_id
        self.position = position
        self.item_id = item_id
        self.pick_point_id = pick_point_id
        self.area_id = area_id
        self.robot_queue = []
        self.picker = None


# 拣货位类
class PickPoint:
    def __init__(self, point_id, position, area_id, item_ids, storage_bin_ids):
        self.point_id = point_id
        self.position = position
        self.area_id = area_id
        self.item_ids = item_ids
        self.storage_bin_ids = storage_bin_ids
        self.robot_queue = []
        self.picker = None
        self.unpicked_items = []

    @property
    def is_idle(self):
        if len(self.robot_queue) > 0 and self.picker is None:
            return True
        else:
            return False


# 机器人类
class Robot(Config):
    def __init__(self, position):
        super().__init__()
        self.parameter = self.parameters["robot"]
        self.position = position
        self.pick_point = None
        self.order = None
        self.item_pick_order = []
        self.state = 'idle'
        self.speed = self.parameter["robot_speed"]
        self.unit_time_cost = None
        self.pick_point_complete_time = 0
        self.move_to_pick_point_time = 0
        self.move_to_depot_time = 0
        self.working_time = 0
        self.run_start_time = None
        self.run_end_time = None
        self.remove = False
        self.rent = None
        self.S_d = self.parameters["warehouse"]["entrance_width"]
        self.S_b = self.parameters["warehouse"]["aisle_width"]
        self.S_l = self.parameters["warehouse"]["shelf_length"]
        self.N_l = self.parameters["warehouse"]["shelf_capacity"]
        self.pick_point_selection_rule = 2

    def assign_order(self, order):
        self.order = order
        self.plan_item_order()

    def plan_item_order(self):
        if self.order is not None:
            self.item_pick_order = [item for item in self.order.items]
        else:
            self.item_pick_order = []

    def next_pick_point(self, pick_points):
        pick_point_ids = [item.pick_point_id for item in self.item_pick_order]
        pick_point_ids = list(set(pick_point_ids))
        if self.pick_point_selection_rule == 1:
            pick_point_ids_sorted = sorted(pick_point_ids,
                                           key=lambda x: (pick_points[x].position[0], pick_points[x].position[1]))
            next_pick_point_id = pick_point_ids_sorted[0]
        elif self.pick_point_selection_rule == 2:
            distances = {point_id: self.distance_between_pick_points(self.position, pick_points[point_id].position) for
                         point_id in pick_point_ids}
            next_pick_point_id = min(distances, key=distances.get)
        elif self.pick_point_selection_rule == 3:
            queue_lengths = {point_id: len(pick_points[point_id].robot_queue) for point_id in pick_point_ids}
            next_pick_point_id = min(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 4:
            queue_lengths = {point_id: len(pick_points[point_id].robot_queue) for point_id in pick_point_ids}
            next_pick_point_id = max(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 5:
            unpicked_counts = {point_id: len(pick_points[point_id].unpicked_items) for point_id in pick_point_ids}
            next_pick_point_id = min(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 6:
            unpicked_counts = {point_id: len(pick_points[point_id].unpicked_items) for point_id in pick_point_ids}
            next_pick_point_id = max(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 7:
            next_pick_point_id = random.choice(pick_point_ids)
        else:
            raise ValueError("拣货位选择规则标识符错误！")
        return pick_points[next_pick_point_id]

    def distance_between_pick_points(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        if x1 == x2:
            return abs(y1 - y2)
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            path2 = (abs(y1 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(
                y2 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(x1 - x2))
            return min(path1, path2)

    def total_run_cost(self, current_time):
        if self.run_end_time is None:
            run_time = current_time - self.run_start_time
            total_cost = run_time * self.unit_time_cost
            return total_cost
        else:
            run_time = self.run_end_time - self.run_start_time
            total_cost = run_time * self.unit_time_cost
            return total_cost

    @property
    def items(self):
        if self.order is not None:
            items = []
            for item in self.order.items:
                if item.pick_point_id == self.pick_point.point_id:
                    items.append(item)
            return items
        return None


# 拣货员类
class Picker(Config):
    def __init__(self, area_id):
        super().__init__()
        self.parameter = self.parameters["picker"]
        self.pick_point = None
        self.position = None
        self.item = None
        self.state = 'idle'
        self.speed = self.parameter["picker_speed"]
        self.area_id = area_id
        self.unit_time_cost = None
        self.storage_bins = []
        self.pick_points = []
        self.working_time = 0
        self.pick_start_time = 0
        self.pick_end_time = 0
        self.remove = False
        self.unit_fire_cost = self.parameter["unit_fire_cost"]
        self.hire_time = None
        self.fire_time = None
        self.rent = None
        self.S_d = self.parameters["warehouse"]["entrance_width"]
        self.S_b = self.parameters["warehouse"]["aisle_width"]
        self.S_l = self.parameters["warehouse"]["shelf_length"]
        self.N_l = self.parameters["warehouse"]["shelf_capacity"]
        self.pick_point_selection_rule = 2

    def total_hire_cost(self, current_time):
        if self.fire_time is None:
            hire_time = current_time - self.hire_time
            total_cost = hire_time * self.unit_time_cost
            return total_cost
        else:
            hire_time = self.fire_time - self.hire_time
            total_cost = hire_time * self.unit_time_cost + self.unit_fire_cost
            return total_cost

    def next_pick_point(self, idle_pick_points_in_area):
        if self.pick_point_selection_rule == 1:
            pick_point_ids_sorted = sorted(idle_pick_points_in_area, key=lambda x: (x.position[0], x.position[1]))
            next_pick_point = pick_point_ids_sorted[0]
            next_pick_point_id = next_pick_point.point_id
        elif self.pick_point_selection_rule == 2:
            distances = {point.point_id: self.distance_between_pick_points(self.position, point.position) for point in
                         idle_pick_points_in_area}
            next_pick_point_id = min(distances, key=distances.get)
        elif self.pick_point_selection_rule == 3:
            queue_lengths = {point.point_id: len(point.robot_queue) for point in idle_pick_points_in_area}
            next_pick_point_id = min(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 4:
            queue_lengths = {point.point_id: len(point.robot_queue) for point in idle_pick_points_in_area}
            next_pick_point_id = max(queue_lengths, key=queue_lengths.get)
        elif self.pick_point_selection_rule == 5:
            unpicked_counts = {point.point_id: len(point.unpicked_items) for point in idle_pick_points_in_area}
            next_pick_point_id = min(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 6:
            unpicked_counts = {point.point_id: len(point.unpicked_items) for point in idle_pick_points_in_area}
            next_pick_point_id = max(unpicked_counts, key=unpicked_counts.get)
        elif self.pick_point_selection_rule == 7:
            next_pick_point = random.choice(idle_pick_points_in_area)
            next_pick_point_id = next_pick_point.point_id
        else:
            raise ValueError("拣货位选择规则标识符错误！")
        next_pick_point = [point for point in idle_pick_points_in_area if point.point_id == next_pick_point_id][0]
        return next_pick_point

    def distance_between_pick_points(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        if x1 == x2:
            return abs(y1 - y2)
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            path2 = (abs(y1 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(
                y2 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(x1 - x2))
            return min(path1, path2)

    @property
    def initial_position(self):
        if not self.pick_points:
            return (0, 0)  # Fallback if no pick points
        x = np.mean([point.position[0] for point in self.pick_points])
        y = np.mean([point.position[1] for point in self.pick_points])
        position = (x, y)
        return position


# ================= 3. 仓库环境类 =================

class WarehouseEnv(gym.Env, Config):
    def __init__(self):
        super().__init__()
        self.parameter = self.parameters["warehouse"]
        self.N_l = self.parameter["shelf_capacity"]
        self.N_s = self.parameter["shelf_levels"]
        self.N_a = self.parameter["area_num"]
        self.N_ai = self.parameter["aisle_num"]
        self.N_w = self.parameter["area_num"] * self.parameter["aisle_num"]
        self.S_l = self.parameter["shelf_length"]
        self.S_w = self.parameter["shelf_width"]
        self.S_b = self.parameter["aisle_width"]
        self.S_d = self.parameter["entrance_width"]
        self.S_a = self.parameter["aisle_width"]
        self.area_dict = {'area{}'.format(i): self.N_ai for i in range(1, self.N_a + 1)}
        self.area_ids = list(self.area_dict.keys())
        self.depot_position = self.parameter["depot_position"]
        self.pack_time = self.parameters["order"]["pack_time"]
        self.total_time = None

        self.pick_points = {}
        self.storage_bins = {}
        self.items = {}
        self.pick_points_area = {area_id: [] for area_id in self.area_ids}
        self.depot_object = Depot(self.depot_position)
        self.create_warehouse_graph()

        self.adjust_pickers_dict = {area_id: None for area_id in self.area_ids}
        self.adjust_robots = None

        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = False
        self.current_time = 0
        self.total_cost_current = 0
        self.total_cost_last = 0

        self.pickers = []
        self.pickers_area = {area_id: [] for area_id in self.area_ids}
        self.pickers_added = []
        self.robots = []
        self.robots_at_depot = []
        self.robots_assigned = []
        self.robots_added = []
        self.orders = []
        self.orders_not_arrived = []
        self.orders_unassigned = []
        self.orders_uncompleted = []
        self.orders_completed = []
        self.orders_arrived = []

    def area_id(self, nw):
        nw_sum = 0
        for area_id, area in self.area_dict.items():
            nw_sum += area
            if nw <= nw_sum:
                return area_id
        print(f"Error: Aisle number {nw} out of range!")
        return None

    def create_warehouse_graph(self):
        for nw in range(1, self.N_w + 1):
            area_id = self.area_id(nw)
            if area_id is None: continue
            for nl in range(1, self.N_l + 1):
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                position = (x, y)

                point_id = f"{nw}-{nl}"
                pick_point = PickPoint(point_id, position, area_id, [], [])
                self.pick_points[point_id] = pick_point
                self.pick_points_area[area_id].append(pick_point)

                for level in range(1, self.N_s + 1):
                    bin_id_left = f"{nw}-{nl}-{level}-left"
                    storage_bin = StorageBin(bin_id_left, position, area_id, None, None)
                    self.storage_bins[bin_id_left] = storage_bin
                    item_id_left = f"{nw}-{nl}-{level}-left-item"
                    item = Item(item_id_left, bin_id_left, position, area_id, None)
                    self.items[item_id_left] = item
                    storage_bin.item_id = item_id_left
                    pick_point.item_ids.append(item_id_left)
                    pick_point.storage_bin_ids.append(bin_id_left)

                    bin_id_right = f"{nw}-{nl}-{level}-right"
                    storage_bin = StorageBin(bin_id_right, position, area_id, None, None)
                    self.storage_bins[bin_id_right] = storage_bin
                    item_id_right = f"{nw}-{nl}-{level}-right-item"
                    item = Item(item_id_right, bin_id_right, position, area_id, None)
                    self.items[item_id_right] = item
                    storage_bin.item_id = item_id_right
                    pick_point.item_ids.append(item_id_right)
                    pick_point.storage_bin_ids.append(bin_id_right)

                    self.storage_bins[bin_id_left].pick_point_id = point_id
                    self.storage_bins[bin_id_right].pick_point_id = point_id
                    self.items[item_id_left].pick_point_id = point_id
                    self.items[item_id_right].pick_point_id = point_id

    def shortest_path_between_pick_points(self, point1, point2):
        x1, y1 = point1.position
        x2, y2 = point2.position
        if x1 == x2:
            return abs(y1 - y2)
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            path2 = (abs(y1 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(
                y2 - (self.S_b * 1.5 + self.N_l * self.S_l)) + abs(x1 - x2))
            return min(path1, path2)

    def adjust_robots_and_pickers(self, n_robots, n_pickers_dict, first_step=False):
        for area_id in self.area_ids:
            if n_pickers_dict[area_id] > 0:
                for i in range(n_pickers_dict[area_id]):
                    picker = Picker(area_id=area_id)
                    picker.pick_points = self.pick_points_area[area_id]
                    picker.position = picker.initial_position
                    picker.hire_time = self.current_time
                    self.pickers_area[area_id].append(picker)
                    self.pickers.append(picker)
                    self.pickers_added.append(picker)
                    if first_step:
                        picker.rent = 'long'
                        picker.unit_time_cost = picker.parameter["long_term_unit_time_cost"]
                    else:
                        picker.rent = 'short'
                        picker.unit_time_cost = picker.parameter["short_term_unit_time_cost"]
            elif n_pickers_dict[area_id] < 0:  # Corrected check from ==0 pass else to <0
                for i in range(abs(n_pickers_dict[area_id])):
                    false_short_pickers = [picker for picker in self.pickers_area[area_id] if
                                           picker.remove is False and picker.rent == 'short']
                    if len(false_short_pickers) <= 0:
                        break
                    else:
                        if len(self.idle_short_rent_pickers[area_id]) > 0:
                            picker = self.idle_short_rent_pickers[area_id].pop(0)
                            picker.remove = True
                            self.pickers_area[area_id].remove(picker)
                            self.pickers.remove(picker)
                            picker.fire_time = self.current_time
                        else:
                            picker = false_short_pickers[0]
                            picker.remove = True

        if n_robots > 0:
            for i in range(n_robots):
                robot = Robot(position=self.depot_position)
                self.robots.append(robot)
                self.robots_at_depot.append(robot)
                robot.run_start_time = self.current_time
                self.robots_added.append(robot)
                if first_step:
                    robot.rent = 'long'
                    robot.unit_time_cost = robot.parameter["long_term_unit_run_cost"]
                else:
                    robot.rent = 'short'
                    robot.unit_time_cost = robot.parameter["short_term_unit_run_cost"]
        elif n_robots < 0:  # Corrected logic
            for i in range(abs(n_robots)):
                false_short_robots = [robot for robot in self.robots if robot.remove is False and robot.rent == 'short']
                if len(false_short_robots) <= 0:
                    break
                else:
                    if len(self.idle_short_rent_robts) > 0:
                        robot = self.idle_short_rent_robts.pop(0)
                        robot.remove = True
                        self.robots.remove(robot)
                        self.robots_at_depot.remove(robot)
                        robot.run_end_time = self.current_time
                    else:
                        robot = false_short_robots[0]
                        robot.remove = True

    def reset(self, orders):
        self.pick_points = {}
        self.storage_bins = {}
        self.items = {}
        self.pick_points_area = {area_id: [] for area_id in self.area_ids}
        self.create_warehouse_graph()

        self.robots = []
        self.robots_at_depot = []
        self.robots_assigned = []
        self.robots_added = []

        self.pickers = []
        self.pickers_added = []
        self.pickers_area = {area_id: [] for area_id in self.area_ids}

        self.orders = copy.deepcopy(orders)
        self.orders_not_arrived = copy.deepcopy(orders)
        self.orders_unassigned = []
        self.orders_uncompleted = []
        self.orders_completed = []
        self.orders_arrived = []

        self.current_time = 0
        self.total_cost_current = 0
        self.total_cost_last = 0

        self.done = False
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

        self.state = self.state_extractor()
        return self.state

    def handle_events(self, current_time):
        """
        处理当前时间点发生的所有离散事件。
        逻辑顺序：订单到达 -> 机器人到达拣货点 -> 拣货完成 -> 机器人回库
        """

        # ================= 1. 新订单到达 =================
        # 使用 while 循环处理可能在同一秒到达的多个订单
        while len(self.orders_not_arrived) > 0 and self.orders_not_arrived[0].arrive_time <= current_time:
            order = self.orders_not_arrived.pop(0)
            self.orders_unassigned.append(order)
            self.orders_uncompleted.append(order)
            self.orders_arrived.append(order)
            # print(f"Time {current_time}: 新订单 {order.order_id} 到达")

        # ================= 2. 机器人移动到拣货点 =================
        for robot in self.robots:
            # 检查是否到达拣货点 (使用差值比较防止浮点数精度问题)
            if abs(robot.move_to_pick_point_time - current_time) < 1e-5:
                # 1. 状态更新
                robot.move_to_pick_point_time = float('inf')  # 事件已处理，重置时间
                pick_point = robot.next_pick_point(self.pick_points)  # 获取当前目标点

                # 2. 物理位置更新
                robot.position = pick_point.position
                robot.pick_point = pick_point
                pick_point.robot_queue.append(robot)  # 加入排队

                # 3. 处理人机协同的时间叠加逻辑 (关键需求)
                # 如果该点已有拣货员且正在忙碌（说明拣货员正在处理队列前面的机器人）
                if pick_point.picker is not None and pick_point.picker.state == 'busy':
                    # 计算该机器人所需拣货时间
                    current_robot_pick_time = sum([item.pick_time for item in robot.items])

                    # 拣货员的结束时间向后顺延（叠加）
                    # 注意：要基于当前已经计划的结束时间累加
                    pick_point.picker.pick_end_time += current_robot_pick_time

                    # 设定该机器人的完成时间 = 拣货员更新后的结束时间
                    robot.pick_point_complete_time = pick_point.picker.pick_end_time
                else:
                    # 如果没有拣货员，机器人只能在此死等（robot.pick_point_complete_time 保持 inf）
                    # 等待 assign_pick_point_to_picker 函数调度空闲拣货员过来
                    pass

        # ================= 3. 拣货完成 (机器人 & 拣货员) =================
        # 注意：先处理机器人离开，再处理拣货员释放

        # 3.1 机器人完成拣货
        for robot in self.robots:
            if abs(robot.pick_point_complete_time - current_time) < 1e-5:
                robot.pick_point_complete_time = float('inf')
                pick_point = robot.pick_point

                # 从队列移除
                if robot in pick_point.robot_queue:
                    pick_point.robot_queue.remove(robot)

                # 更新订单和商品状态
                items_to_remove = []  # 暂存要移除的商品
                for item in robot.items:
                    robot.order.picked_items.append(item)
                    if item in robot.order.unpicked_items:
                        robot.order.unpicked_items.remove(item)
                    if item in robot.item_pick_order:
                        robot.item_pick_order.remove(item)

                # 规划下一步行动
                if len(robot.item_pick_order) > 0:
                    # 还有商品没拣，去下一个点
                    next_pick_point = robot.next_pick_point(self.pick_points)
                    dist = self.shortest_path_between_pick_points(robot, next_pick_point)
                    move_time = dist / robot.speed
                    robot.move_to_pick_point_time = current_time + move_time
                else:
                    # 订单完成，回 Depot (包含打包时间)
                    dist = self.shortest_path_between_pick_points(robot, self.depot_object)
                    move_time = dist / robot.speed
                    robot.move_to_depot_time = current_time + move_time + self.pack_time

        # 3.2 拣货员完成任务
        # 使用副本遍历，防止在移除列表元素时出错
        for picker in self.pickers[:]:
            if abs(picker.pick_end_time - current_time) < 1e-5:
                picker.pick_end_time = float('inf')
                picker.state = 'idle'

                # 解绑拣货位
                if picker.pick_point:
                    picker.pick_point.picker = None
                    picker.pick_point = None

                # 处理延迟移除逻辑 (Requirement 2)
                if picker.remove:
                    self.pickers_area[picker.area_id].remove(picker)
                    self.pickers.remove(picker)
                    picker.fire_time = current_time
                    # print(f"Time {current_time}: 拣货员 {picker} 被移除")

        # ================= 4. 机器人回库 (订单完成) =================
        for robot in self.robots[:]:
            if abs(robot.move_to_depot_time - current_time) < 1e-5:
                robot.move_to_depot_time = float('inf')
                robot.state = 'idle'
                robot.position = self.depot_position

                # 订单结算
                if robot.order:
                    if robot.order in self.orders_uncompleted:
                        self.orders_uncompleted.remove(robot.order)
                    self.orders_completed.append(robot.order)
                    robot.order.complete_time = current_time
                    robot.order = None  # 重置关联

                # 处理延迟移除逻辑 (Requirement 2)
                if robot.remove:
                    self.robots.remove(robot)
                    self.robots_at_depot.remove(robot)
                    robot.run_end_time = current_time
                    # print(f"Time {current_time}: 机器人 {robot} 被移除")
                else:
                    # 如果未被移除且变为空闲，立即尝试分配新订单
                    # 避免等到下一个时间步才分配
                    self.robots_at_depot.append(robot)  # 确保在depot列表
                    # 尝试触发一次分配逻辑
                    # 注意：这里不能直接递归调用，因为可能造成逻辑混乱，
                    # 最好由主循环的 assign_order_to_robot 处理
                    pass

    def step(self, action, first_step=False, pattern=None):
        self.action = np.round(action).astype(int)
        self.adjust_robots = self.action[0]
        if len(self.robots) + self.adjust_robots <= 0:
            self.adjust_robots = -len(self.robots) + 1

        self.adjust_pickers_dict = {area_id: self.action[i] for i, area_id in enumerate(self.area_ids, start=1)}
        for area_id in self.area_ids:
            if len(self.pickers_area[area_id]) + self.adjust_pickers_dict[area_id] <= 0:
                self.adjust_pickers_dict[area_id] = -len(self.pickers_area[area_id]) + 1

        self.adjust_robots_and_pickers(self.adjust_robots, self.adjust_pickers_dict, first_step)
        if pattern == 'long':
            epoch_time = 8 * 3600 * 30
        else:
            epoch_time = 8 * 3600

        end_time = self.current_time + epoch_time

        while self.current_time < end_time:
            # 如果所有订单都处理完了，且没有新订单要来，可以选择提前跳出（取决于业务逻辑，是24小时待命还是做完就休）
            if len(self.orders_not_arrived) == 0 and len(self.orders_uncompleted) == 0:
                break

            # 1. 尝试分配任务 (给空闲的机器人/拣货员找活干)
            self.assign_order_to_robot()
            for area_id in self.area_ids:
                self.assign_pick_point_to_picker(area_id)

            # 2. 计算下一个事件发生的时间
            # 收集所有可能的未来时间点
            future_events = []
            if self.orders_not_arrived:
                future_events.append(self.orders_not_arrived[0].arrive_time)

            # 收集所有忙碌实体的完成时间
            for p in self.pickers:
                if p.state == 'busy': future_events.append(p.pick_end_time)
            for r in self.robots:
                if r.state == 'busy':
                    # 机器人有三个阶段的忙碌时间，取当前正在进行的那个
                    times = [r.move_to_pick_point_time, r.pick_point_complete_time, r.move_to_depot_time]
                    # 过滤掉无穷大和已经过去的时间
                    future_events.extend([t for t in times if t != float('inf') and t > self.current_time])

            # 3. 确定步进时间
            valid_events = [t for t in future_events if t > self.current_time]

            if not valid_events:
                # 没有未来事件了（所有订单处理完，所有人都空闲）
                # 直接跳到结束时间
                self.current_time = end_time
                break

            next_event_time = min(valid_events)

            # 如果下一个事件超过了今天结束时间，则只运行到今天结束
            if next_event_time > end_time:
                self.current_time = end_time
                break

            # 更新当前时间
            self.current_time = next_event_time

            # 4. 处理在该时间点发生的所有事件
            self.handle_events(self.current_time)  # 将事件处理逻辑封装

        # 5. 检查是否结束
        if self.current_time >= self.total_time:
            self.done = True
        # 6. 提取下一个状态 & 计算奖励
        self.state = self.state_extractor()
        # 7. 计算总成本
        total_delay_cost = sum([order.total_delay_cost(self.current_time) for order in self.orders_arrived])
        total_run_cost = sum([robot.total_run_cost(self.current_time) for robot in self.robots_added])
        total_hire_cost = sum([picker.total_hire_cost(self.current_time) for picker in self.pickers_added])
        self.total_cost_current = total_delay_cost + total_run_cost + total_hire_cost
        # 计算奖励
        self.reward = self.compute_reward()
        self.total_cost_last = self.total_cost_current

        return self.state, self.reward, self.done

    def assign_order_to_robot(self):
        while len(self.orders_unassigned) > 0 and len(self.idle_robots) > 0:
            robot = self.idle_robots.pop(0)
            order = self.orders_unassigned.pop(0)
            robot.assign_order(order)
            next_pick_point = robot.next_pick_point(self.pick_points)
            shortest_path_length = self.shortest_path_between_pick_points(robot, next_pick_point)
            move_time = shortest_path_length / robot.speed
            robot.state = 'busy'
            robot.working_time += move_time
            robot.move_to_pick_point_time = self.current_time + move_time

    def assign_pick_point_to_picker(self, area_id):
        while len(self.idle_pickers[area_id]) > 0 and len(self.idle_pick_points[area_id]) > 0:
            idle_pickers_in_area = self.idle_pickers[area_id]
            idle_pick_points_in_area = self.idle_pick_points[area_id]
            picker = random.choice(idle_pickers_in_area)
            pick_point = picker.next_pick_point(idle_pick_points_in_area)

            picker.pick_point = pick_point
            pick_point.picker = picker

            shortest_path_length = self.shortest_path_between_pick_points(picker, pick_point)
            move_time = shortest_path_length / picker.speed

            picker.state = 'busy'
            picker.working_time += move_time
            picker.pick_start_time = self.current_time + move_time
            picker.pick_end_time = picker.pick_start_time

            # Sort robots in queue? Assuming FIFO
            for robot in pick_point.robot_queue:
                picker.pick_end_time += sum([i.pick_time for i in robot.items])
                robot.pick_point_complete_time = picker.pick_end_time

            picker.position = pick_point.position

    def state_extractor(self):
        robot_queue_list = [len(point.robot_queue) for point in self.pick_points.values()]
        robot_queue_list = np.array(robot_queue_list).reshape((self.N_w, self.N_l))
        picker_list = [0 if point.picker is None else 1 for point in self.pick_points.values()]
        picker_list = np.array(picker_list).reshape((self.N_w, self.N_l))
        unpicked_items_list = self.pick_point_unpicked_items
        unpicked_items_list = np.array(unpicked_items_list).reshape((self.N_w, self.N_l))
        n_robots = len([robot for robot in self.robots if robot.remove is False])
        n_pickers_area = [len(self.pickers_area[area_id]) for area_id in self.area_ids]

        self.state = {'robot_queue_list': robot_queue_list, 'picker_list': picker_list,
                      'unpicked_items_list': unpicked_items_list,
                      'n_robots': n_robots, 'n_pickers_area': n_pickers_area}
        return self.state

    def compute_reward(self):
        return self.total_cost_last - self.total_cost_current

    @property
    def idle_robots(self):
        return [robot for robot in self.robots if robot.state == 'idle']

    @property
    def idle_short_rent_robts(self):
        return [robot for robot in self.robots if robot.state == 'idle' and robot.rent == 'short']

    @property
    def idle_pickers(self):
        idle_pickers_area = {area_id: [picker for picker in self.pickers_area[area_id]
                                       if picker.state == 'idle'] for area_id in self.area_ids}
        return idle_pickers_area

    @property
    def idle_short_rent_pickers(self):
        idle_pickers_area = {area_id: [picker for picker in self.pickers_area[area_id]
                                       if picker.state == 'idle' and picker.rent == 'short'] for area_id in
                             self.area_ids}
        return idle_pickers_area

    @property
    def idle_pick_points(self):
        # 每个区域待分配拣货员的拣货位列表字典
        # 修正：这里必须使用 self.pick_points_area (存放 PickPoint 对象)，而不是 self.pickers_area
        idle_pick_points_area = {
            area_id: [point for point in self.pick_points_area[area_id] if point.is_idle]
            for area_id in self.area_ids
        }
        return idle_pick_points_area

    @property
    def pick_point_unpicked_items(self):
        for point in self.pick_points.values():
            point.unpicked_items = []
        for order in self.orders_uncompleted:
            for item in order.unpicked_items:
                pick_point_id = item.pick_point_id
                self.pick_points[pick_point_id].unpicked_items.append(item)
        unpicked_items_list = [len(point.unpicked_items) for point in self.pick_points.values()]
        return unpicked_items_list


if __name__ == "__main__":
    # 基于仓库中的商品创建一个月内的订单对象，每个订单包含多个商品，订单到达时间服从泊松分布，仿真周期设置为一个月
    # 总秒数
    total_seconds = (8 * 3600) * 30  # 30天
    num_items = 6

    # 初始化仓库环境
    warehouse = WarehouseEnv()
    # 输出储货位总数
    print('储货位总数：', len(warehouse.storage_bins))
    # 输出商品总数
    print('商品总数：', len(warehouse.items))

    # 订单数据读取
    file_order = 'D:\\Python project\\DRL_Warehouse\\data\\instances'
    with open(file_order + "\\orders_{}.pkl".format(num_items), "rb") as f:
        orders = pickle.load(f)  # 读取订单数据

    warehouse.total_time = total_seconds

    for epoch in range(1):
        orders_object = copy.deepcopy(orders)
        print('订单数量：', len(orders_object))
        warehouse.reset(orders_object)
        total_reward = 0
        while not warehouse.done:
            n_robot = random.randint(4, 4)
            n_picker = random.randint(1, 1)
            action = [n_robot, n_picker, n_picker, n_picker]

            state, reward, done = warehouse.step(action, first_step=(epoch == 0 and warehouse.current_time == 0), pattern='long')
            total_reward += reward
            # print(f"Time: {warehouse.current_time}, Robots: {len(warehouse.robots)}, Pickers: {len(warehouse.pickers)}")

        print(f"Total reward: {total_reward}")