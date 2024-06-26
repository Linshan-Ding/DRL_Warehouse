"""
智能仓库人机协同拣选系统仿真环境
"""
import numpy as np
import random
import time
import copy
import os
import sys
import math


class Order:
    def __init__(self, order_id, items, arrive_time=0):
        self.order_id = order_id  # 订单的编号
        self.items = items  # 订单中的商品列表
        self.arrive_time = arrive_time  # 订单到达时间
        self.complete_time = None  # 订单拣选完成时间
        # 单位拣选时间成本
        self.unit_time_cost = 1
        # 订单中的未拣选完成的商品列表
        self.unpicked_items = copy.deepcopy(self.items)
        # 订单中的已拣选完成的商品列表
        self.picked_items = []


class Item:
    def __init__(self, item_id, bin_id, position, area_id, pick_point_id):
        self.item_id = item_id  # 商品的编号
        self.bin_id = bin_id  # 商品所在的储货位编号
        self.position = position  # 商品所在的位置
        self.area_id = area_id  # 商品所在的区域编号
        self.pick_point_id = pick_point_id  # 商品所属拣货位的编号
        self.pick_time = 1  # 拣选时间


# 储货位类
class StorageBin:
    def __init__(self, bin_id, position, area_id, item_id, pick_point_id):
        self.bin_id = bin_id  # 储货位的编号
        self.position = position  # 储货位的位置
        self.item_id = item_id  # 储货位中的商品编号
        self.pick_point_id = pick_point_id  # 储货位所属拣货位的编号
        # 储货位所属区域的编号
        self.area_id = area_id
        # 当前储货位的机器人对象队列
        self.robot_queue = []
        # 当前储货位的拣货员对象
        self.picker = None


# 拣货位类
class PickPoint:
    def __init__(self, point_id, position, area_id, item_ids, storage_bin_ids):
        self.point_id = point_id  # 拣货位的编号
        self.position = position  # 拣货位的位置
        self.area_id = area_id  # 拣货位所属区域的编号
        self.item_ids = item_ids  # 拣货位中的商品编号列表
        self.storage_bin_ids = storage_bin_ids  # 拣货位对应的储货位编号列表
        # 拣货位的机器人对象队列
        self.robot_queue = []
        # 拣货位的拣货员对象
        self.picker = None
        # 拣货位的未拣货商品列表
        self.unpicked_items = []

    # 监测拣货位置是否待分配拣货员
    @property
    def is_idle(self):
        # 如果拣货位上没有拣货员且机器人队列中有机器人，则返回True
        if len(self.robot_queue) > 0 and self.picker is None:
            return True
        # 如果拣货位上有拣货员，则返回False
        else:
            return False


class Robot:
    def __init__(self, robot_id, position):
        self.robot_id = robot_id  # 机器人的编号
        self.position = position  # 机器人的位置
        self.order = None  # 机器人关联的订单
        self.path = []  # 机器人的路径
        self.state = 'idle'  # 机器人所处状态：'idle', 'busy', 'moving'
        self.speed = 2  # 机器人移动速度
        # 机器人工作单位时间成本
        self.unit_time_cost = 1

    def assign_order(self, order):
        self.order = order
        self.plan_path()

    def plan_path(self):
        # 订单中的商品拣选顺序规划
        if self.order:
            self.path = [item.position for item in self.order.items]
        else:
            self.path = []

    def move_to_next_position(self):
        if self.path:
            self.position = self.path.pop(0)

    # 从仓库中移除该机器人，重置该机器人到初始状态
    def reset(self):
        self.position = None
        self.position = None
        self.order = None
        self.path = []
        self.state = 'idle'


class Picker:
    def __init__(self, picker_id, area_id):
        self.picker_id = picker_id  # 拣货员的编号
        self.position = None  # 拣货员当前位置
        self.item = None  # 拣货员当前处理的商品
        self.state = 'idle'  # State can be 'idle', 'busy', 'moving'
        self.speed = 2  # 拣货员移动速度
        self.area_id = area_id  # 拣货员所在区域的编号
        # 拣货员工作单位时间成本
        self.unit_time_cost = 1
        # 拣货员负责的储货位列表
        self.storage_bins = []
        # 拣货员负责的拣货位列表
        self.pick_points = []
        # 拣货员当天的工作时间：包括拣货时间和移动时间
        self.working_time = 0
        # 拣货员下个要移动到的位置
        self.next_position = None
        # 拣货员当前位于的拣货位
        self.pick_point = None

    # 根据负责的拣货位列表中的拣货位的坐标计算拣货员的初始位置（取各拣货位的坐标均值）
    @ property
    def initial_position(self):
        x = np.mean([point.position[0] for point in self.pick_points])
        y = np.mean([point.position[1] for point in self.pick_points])
        position = (x, y)
        return position

    # 从仓库中移除该机器人，重置该机器人到初始状态
    def reset(self):
        self.position = None
        self.item = None
        self.state = 'idle'
        self.working_time = 0
        self.storage_bins = []
        self.pick_points = []


# 仓库环境类：
# 包括机器人、拣货员、拣货位、储货位和商品；
# 步进函数step()实现仓库环境的仿真;
# 动作为每间隔24个小时调整每个区域的拣货员和仓库中总的机器人的数量；
class WarehouseEnv:
    def __init__(self, N_l, N_w, S_l, S_w, S_b, S_d, S_a, depot_position):
        # 仓库环境参数
        self.N_l = N_l  # 单个货架中储货位的数量
        self.N_w = N_w  # 巷道的数量
        self.S_l = S_l  # 储货位的长度
        self.S_w = S_w  # 储货位的宽度
        self.S_b = S_b  # 底部通道的宽度
        self.S_d = S_d  # 仓库的出入口处的宽度
        self.S_a = S_a  # 巷道的宽度
        self.N_w_area = 3  # 仓库中每个区域包含的巷道数量
        self.depot_position = depot_position  # 机器人的起始位置
        # 仓库固定属性
        self.pick_points = {}  # 拣货位字典
        self.storage_bins = {}  # 储货位字典
        self.items = {}  # 商品字典
        self.area_ids = []  # 仓库区域编号列表
        self.pick_points_area = {}  # 每个区域的拣货位列表字典
        # 构建仓库图
        self.create_warehouse_graph()

        # 为仓库调整机器人数量和每个区域调整拣货员数量
        self.adjust_pickers_dict = {area_id: None for area_id in self.area_ids}  # 每个区域初始拣货员数量
        self.adjust_robots = None  # 仓库中初始机器人数量

        # 仓库强化学习环境属性
        self.state = None  # 当前状态
        self.action = None  # 当前动作
        self.next_state = None  # 下一个状态
        self.reward = None  # 当前奖励
        self.total_reward = 0  # 累计奖励
        self.done = False  # 是否结束标志
        self.current_time = 0  # 当前时间

        # 仓库中的机器人对象信息
        self.pickers = []  # 拣货员列表
        self.pickers_list = []  # 截止目前实例化的拣货员列表
        self.pickers_area = {area_id: [] for area_id in self.area_ids}  # 每个区域的拣货员列表字典
        # 仓库中的拣货员对象信息
        self.robots = []  # 机器人列表
        self.robots_list = []  # 截止目前实例化的机器人对象列表
        self.robots_at_depot = []  # depot_position位置的机器人列表
        self.robots_assigned = []  # 已分配订单的机器人列表
        # 仓库仿真环境时钟和订单对象属性
        self.orders = []  # 已到达订单列表
        self.orders_completed = []  # 已拣选完成订单列表
        self.orders_uncompleted = []  # 未拣选完成的订单列表

    def create_warehouse_graph(self):
        # 创建仓库图, 包括货架、巷道、储货位和商品
        for nw in range(1, self.N_w + 1):
            # 计算该巷道所处的区域数字标识
            id = math.ceil(nw / self.N_w_area)
            area_id = "area" + str(id)
            self.pick_points_area[area_id] = []  # 初始化每个区域的拣货位列表
            self.area_ids.append(area_id)  # 将区域编号加入到区域编号列表中
            for nl in range(1, self.N_l + 1):
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                # 计算拣货位的位置
                position = (x, y)

                # 创建该拣货位左侧储货位对象
                bin_id_left = f"{nw}-{nl}-left"
                storage_bin = StorageBin(bin_id_left, position, area_id, None, None)
                self.storage_bins[bin_id_left] = storage_bin
                # 创建该储货位存储的商品对象
                item_id_left = f"{nw}-{nl}-left-item"
                item = Item(item_id_left, bin_id_left, position, area_id, None)
                self.items[item_id_left] = item
                # 将商品放入储货位
                storage_bin.item_id = item_id_left

                # 创建该拣货位右侧储货位对象
                bin_id_right = f"{nw}-{nl}-right"
                storage_bin = StorageBin(bin_id_right, position, area_id, None, None)
                self.storage_bins[bin_id_right] = storage_bin
                # 创建该储货位存储的商品对象
                item_id_right = f"{nw}-{nl}-right-item"
                item = Item(item_id_right, bin_id_right, position, area_id, None)
                self.items[item_id_right] = item
                # 将商品放入储货位
                storage_bin.item_id = item_id_right

                # 创建拣货位对象
                point_id = f"{nw}-{nl}"
                pick_point = PickPoint(point_id, position, area_id, [item_id_left, item_id_right], [bin_id_left, bin_id_right])
                self.pick_points[point_id] = pick_point  # 将拣货位加入到拣货位字典中
                self.pick_points_area[area_id].append(pick_point)  # 将拣货位加入到对应区域的拣货位列表中

                # 将拣货位和储货位+商品关联
                self.storage_bins[bin_id_left].pick_point_id = point_id
                self.storage_bins[bin_id_right].pick_point_id = point_id
                self.items[item_id_left].pick_point_id = point_id
                self.items[item_id_right].pick_point_id = point_id

    # 两个拣货位之间的最短路径长度（若不在一个巷道则需要从上部或下部绕过储货位）
    def shortest_path_between_pick_points(self, point1, point2):
        x1, y1 = point1.position
        x2, y2 = point2.position
        # 如果两个拣货位在同一巷道，则返回两个拣货位之间的直线路径长度
        if x1 == x2:
            return abs(y1 - y2)
        # 计算从上部绕过和从下部绕过的路径，选择最短路径，并返回路径长度
        else:
            path1 = abs(y1 - self.S_b) + abs(y2 - self.S_b) + abs(x1 - x2)
            path2 = abs(y1 - (self.S_b + self.N_l * self.S_l)) + abs(y2 - (self.S_b + self.N_l * self.S_l)) + abs(x1 - x2)
            return min(path1, path2)

    def adjust_robots_and_pickers(self, n_robots, n_pickers_dict):
        """为仓库中添加初始化的机器人和每个区域的拣货员"""
        # 更新每个区域中拣货员数量
        for area_id in self.area_ids:
            if n_pickers_dict[area_id] >= 1:  # 如果该区域添加的拣货员数量大于等于1
                for i in range(n_pickers_dict[area_id]):
                    picker = Picker(picker_id=f"picker-{area_id}-{i}", area_id=area_id)
                    picker.pick_points = self.pick_points_area[area_id]  # 拣货员负责的拣货位列表
                    picker.position = picker.initial_position  # 根据负责的拣货位列表中的拣货位的坐标计算拣货员的初始位置
                    self.pickers_area[area_id].append(picker)  # 将拣货员加入到对应区域的拣货员列表中
                    self.pickers_list.append(picker)  # 将拣货员加入到拣货员列表中
                    self.pickers.append(picker)  # 将拣货员加入到拣货员列表中
            elif n_pickers_dict[area_id] == 0:  # 如果该区域添加的拣货员数量等于0
                pass
            else:  # 如果该区域添加的拣货员数量小于0
                # 从各区域空闲的拣货员中移除多余的拣货员
                for i in range(abs(n_pickers_dict[area_id])):
                    picker = self.idle_pickers[area_id].pop(0)  # 从各区域空闲的拣货员中移除拣货员
                    self.pickers_area[area_id].remove(picker)  # 从对应区域的拣货员列表中移除拣货员
                    self.pickers.remove(picker)  # 从拣货员列表中移除拣货员

        # 更新机器人数量
        if n_robots >= 1:
            # 实例化机器人对象并添加到仓库中
            for i in range(n_robots):
                robot = Robot(robot_id=f"robot-{i}", position=self.depot_position)
                self.robots_list.append(robot)  # 将机器人加入到截止目前实例化的机器人对象列表中
                self.robots.append(robot)  # 将机器人加入到机器人列表中
                self.robots_at_depot.append(robot)  # 将机器人加入到depot_position位置的机器人列表中
        elif n_robots == 0:
            pass
        else:
            # 从depot_position位置移除多余的机器人
            for i in range(abs(n_robots)):
                robot = self.robots_at_depot.pop(0)  # 从depot_position位置移除机器人
                self.robots.remove(robot)  # 从机器人列表中移除机器人

    def reset(self):
        """重置仓库环境"""
        # 重置仓库中的机器人和拣货员对象信息
        self.robots = []  # 机器人列表
        self.pickers = []  # 拣货员列表
        self.pickers_list = []  # 截止目前实例化的拣货员列表
        self.robots_list = []  # 截止目前实例化的机器人对象列表
        self.pickers_area = {area_id: [] for area_id in self.area_ids}  # 每个区域的拣货员列表字典
        # 重置仓库强化学习环境属性
        self.state = None  # 当前状态
        self.action = None  # 当前动作
        self.next_state = None  # 下一个状态
        self.reward = None  # 当前奖励
        self.total_reward = 0  # 累计奖励
        self.done = False  # 是否结束标志
        # 重置仓库仿真环境时钟和订单对象属性
        self.current_time = 0  # 当前时间
        self.orders = []  # 已到达订单列表
        self.orders_completed = []  # 已拣选完成订单列表
        self.orders_uncompleted = []  # 未拣选完成的订单列表
        self.robots_at_depot = []  # depot_position位置的机器人列表
        # 提取初始状态
        self.state = self.state_extractor()
        return self.state

    def step(self, action):
        """
        仓库环境的仿真步进函数：每个决策点执行一次step()函数。
        决策点：时钟移动到每天的开始时刻时。
        离散点：新订单到达时刻、拣货员空闲时刻、机器人移动到拣货点时刻，机器人空闲时刻。
        action: 每天的开始时刻机器人和各区域内拣货员的调整值。
        """
        self.adjust_robots = action[0]  # 动作调整的机器人数量
        self.adjust_pickers_dict = {area_id: action[i] for i, area_id in enumerate(self.area_ids, start=1)}  # 动作调整的每个区域的拣货员数量
        # 执行动作，调整仓库中的机器人和拣货员数量
        self.adjust_robots_and_pickers(self.adjust_robots, self.adjust_pickers_dict)
        # 一天的仿真时间
        one_day = 24 * 3600
        # 初始时间
        start_time = self.current_time
        # 结束时间
        end_time = self.current_time + one_day
        while self.current_time < end_time:  # 当前时间小于结束时间时
            # 判断是否移动时钟
            # 若当前离散点[新订单到达时刻，拣货员拣货完成时刻，机器人移动到拣货点时刻，机器人空闲时刻]中的最小值大于当前时间时，则移动时钟到下一个离散点时刻
            # 若当前离散点某个区域同时存在空闲拣货员和待分配拣货员的拣货位时，则时钟保持不动。并执行拣货员分配动作，直到每个区域不同时存在空闲拣货员和待分配拣货员的拣货位时
            pass
        pass

    def state_extractor(self):
        """提取仓库的当前状态"""
        # 每个拣货位的机器人数量列表
        robot_queue_list = [len(point.robot_queue) for point in self.pick_points.values()]
        # 每个拣货位是否有拣货员拣货员，有的话为1，没有的话为0
        picker_list = [0 if point.picker is None else 1 for point in self.pick_points.values()]
        # 每个拣货位未拣货商品数量
        unpicked_items_list = self.pick_point_unpicked_items
        # depot_position位置的机器人数量
        n_robots_at_depot = len(self.robots_at_depot)
        # 连接所有状态特征，并转为numpy，提取为状态向量
        state = np.array(robot_queue_list + picker_list + unpicked_items_list + [n_robots_at_depot])
        return state

    def compute_reward(self):
        """计算当前奖励"""
        pass

    # 当前离散点空闲机器人列表
    @ property
    def idle_robots(self):
        return [robot for robot in self.robots if robot.state == 'idle']

    # 当前离散点每个区域的空闲拣货员列表
    @ property
    def idle_pickers(self):
        # 每个区域的空闲拣货员列表字典
        idle_pickers_area = {area_id: [picker for picker in self.pickers_area[area_id] if picker.state == 'idle'] for area_id in self.area_ids}
        return idle_pickers_area

    # 当前离散点每个区域待分配拣货员的拣货位列表
    @ property
    def idle_pick_points(self):
        # 每个区域待分配拣货员的拣货位列表字典
        idle_pick_points_area = {area_id: [point for point in self.pick_points_area[area_id] if point.is_idle] for area_id in self.area_ids}
        return idle_pick_points_area

    # 当前离散点每个拣货位未拣货商品数量（基于未拣货完成订单中的未拣货完成商品计算对应拣货位的未拣货商品数量）
    @ property
    def pick_point_unpicked_items(self):
        # 重置拣货位中的未拣货商品列表
        for point in self.pick_points.values():
            point.unpicked_items = []
        # 计算拣货位中的未拣货商品数量
        for order in self.orders_uncompleted:
            for item in order.unpicked_items:
                pick_point_id = item.pick_point_id
                self.pick_points[pick_point_id].unpicked_items.append(item)
        # 每个拣货位未拣货商品数量列表
        unpicked_items_list = [len(point.unpicked_items) for point in self.pick_points.values()]
        return unpicked_items_list


if __name__ == "__main__":
    # 初始化仓库环境
    N_l = 10  # 单个货架中储货位的数量
    N_w = 6  # 巷道的数量
    S_l = 1  # 储货位的长度
    S_w = 1  # 储货位的宽度
    S_b = 2  # 底部通道的宽度
    S_d = 2  # 仓库的出入口处的宽度
    S_a = 2  # 巷道的宽度
    depot_position = (0, 0)  # 机器人的起始位置

    # 初始化仓库环境
    warehouse = WarehouseEnv(N_l, N_w, S_l, S_w, S_b, S_d, S_a, depot_position)

    # 基于仓库中的商品创建一个月内的订单对象，每个订单包含多个商品，订单到达时间服从泊松分布，仿真周期设置为一个月
    # 一个月的总秒数
    total_seconds = 30 * 24 * 3600
    # 一个月内的订单列表
    orders = []
    # 订单编号
    order_id = 1
    # 订单到达时间
    arrival_time = 0
    while arrival_time < total_seconds:
        # 订单中的商品列表
        items = []
        # 订单中的商品数量
        n_items = random.randint(1, 10)
        for i in range(n_items):
            # 随机选择一个商品
            item_id = random.choice(list(warehouse.items.keys()))
            items.append(warehouse.items[item_id])
        # 创建订单对象
        order = Order(order_id, items, arrival_time)
        orders.append(order)
        # 生成下一个订单到达时间
        arrival_time += random.expovariate(1 / 3600)
        order_id += 1
    print(f"Total number of orders: {len(orders)}")

    # 基于上述一个月内的订单数据和仓库环境数据，实现仓库环境的仿真

