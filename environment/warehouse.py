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


# 起始点类
class Depot:
    def __init__(self, position):
        self.position = position  # 起始点的位置


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
        # 如果拣货位上未分配拣货员且机器人队列中有机器人，则返回True
        if len(self.robot_queue) > 0 and self.picker is None:
            return True
        # 如果拣货位上有拣货员，则返回False
        else:
            return False


class Robot:
    def __init__(self, position):
        self.position = position  # 机器人的位置
        self.order = None  # 机器人关联的订单
        self.item_pick_order = []  # 机器人的商品拣选顺序
        self.state = 'idle'  # 机器人所处状态：'idle', 'busy'
        self.speed = 2  # 机器人移动速度
        self.unit_time_cost = 1  # 机器人工作单位时间成本
        self.item = None  # 机器人待拣选或正在拣选的商品
        self.item_pick_complete_time = 0  # 机器人的当前商品拣货完成时间
        # 机器人移动到拣货位的时间
        self.move_to_pick_point_time = 0
        # 机器人移动到depot_position的时间
        self.move_to_depot_time = 0

    def assign_order(self, order):
        """为机器人分配订单"""
        self.order = order
        self.plan_item_order()

    def plan_item_order(self):
        """订单中的商品对象拣选顺序规划"""
        if self.order is not None:
            self.item_pick_order = [item for item in self.order.items]
        else:
            self.item_pick_order = []


class Picker:
    def __init__(self, area_id):
        self.pick_point = None  # 拣货员当前拣货位
        self.position = None  # 拣货员的位置
        self.item = None  # 拣货员待拣选或正在拣选的商品
        self.state = 'idle'  # 拣货员状态：'idle', 'busy'
        self.speed = 2  # 拣货员移动速度
        self.area_id = area_id  # 拣货员所在区域的编号
        self.unit_time_cost = 1  # 拣货员工作单位时间成本
        self.storage_bins = []  # 拣货员负责的储货位列表
        self.pick_points = []  # 拣货员负责的拣货位列表
        self.working_time = 0  # 拣货员工作时间
        self.pick_start_time = 0  # 拣货员在当前拣货位拣货开始时间
        self.pick_end_time = 0  # 拣货员在当前拣货位拣货结束时间

    # 根据负责的拣货位列表中的拣货位的坐标计算拣货员的初始位置（取各拣货位的坐标均值）
    @ property
    def initial_position(self):
        x = np.mean([point.position[0] for point in self.pick_points])
        y = np.mean([point.position[1] for point in self.pick_points])
        position = (x, y)
        return position


# -------------------------仓库环境类---------------------------
# 包括机器人、拣货员、拣货位、储货位和商品
# 步进函数step()实现仓库环境的仿真
# 动作为每间隔24个小时调整每个区域的拣货员和仓库中总的机器人的数量
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
        self.depot_object = Depot(depot_position)  # 仓库起始点对象
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

        # 仓库中的拣货员对象信息
        self.pickers = []  # 拣货员列表
        self.pickers_area = {area_id: [] for area_id in self.area_ids}  # 每个区域的拣货员列表字典
        # 仓库中的机器人对象信息
        self.robots = []  # 机器人列表
        self.robots_at_depot = []  # depot_position位置的机器人列表
        self.robots_assigned = []  # 已分配订单的机器人列表
        # 仓库中的订单对象信息
        self.orders = []  # 整个仿真过程所有订单对象列表
        self.orders_not_arrived = []  # 未到达的订单对象列表
        self.orders_unassigned = []  # 已到达但未分配机器人的订单对象列表
        self.orders_uncompleted = []  # 未拣选完成的订单对象列表

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

    # 两个拣货位之间的最短路径长度（若不在一个巷道，则需要从上部或下部绕过储货位）
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
            # 如果该区域添加的拣货员数量大于0
            if n_pickers_dict[area_id] > 0:
                for i in range(n_pickers_dict[area_id]):
                    picker = Picker(area_id=area_id)
                    picker.pick_points = self.pick_points_area[area_id]  # 拣货员负责的拣货位列表
                    picker.position = picker.initial_position  # 根据负责的拣货位列表中的拣货位的坐标计算拣货员的初始位置
                    self.pickers_area[area_id].append(picker)  # 将拣货员加入到对应区域的拣货员列表中
                    self.pickers.append(picker)  # 将拣货员加入到拣货员列表中
            # 如果该区域添加的拣货员数量等于0
            elif n_pickers_dict[area_id] == 0:
                pass
            # 如果该区域添加的拣货员数量小于0
            else:
                # 从各区域空闲的拣货员中移除多余的拣货员
                for i in range(abs(n_pickers_dict[area_id])):
                    picker = self.idle_pickers[area_id].pop(0)  # 从各区域空闲的拣货员中移除拣货员
                    self.pickers_area[area_id].remove(picker)  # 从对应区域的拣货员列表中移除拣货员
                    self.pickers.remove(picker)  # 从拣货员列表中移除拣货员

        # 更新机器人数量
        if n_robots > 0:
            # 实例化机器人对象并添加到仓库中
            for i in range(n_robots):
                robot = Robot(position=self.depot_position)
                self.robots.append(robot)  # 将机器人加入到机器人列表中
                self.robots_at_depot.append(robot)  # 将机器人加入到depot_position位置的机器人列表中
        elif n_robots == 0:
            pass
        else:
            # 从depot_position位置移除多余的机器人
            for i in range(abs(n_robots)):
                robot = self.robots_at_depot.pop(0)  # 从depot_position位置移除机器人
                self.robots.remove(robot)  # 从机器人列表中移除机器人

    def reset(self, orders):
        """重置仓库环境"""
        # 重置仓库中的机器人和拣货员对象信息
        self.robots = []  # 机器人列表
        self.pickers = []  # 拣货员列表
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
        self.orders = orders  # 整个仿真过程所有订单对象列表
        self.orders_not_arrived = orders  # 未到达的订单对象列表
        self.orders_unassigned = []  # 未分配机器人的订单对象列表
        self.orders_uncompleted = []  # 未拣选完成的订单对象列表
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
        # 当前step开始时间
        start_time = self.current_time
        # 结束时间
        end_time = self.current_time + one_day

        # 仿真该step: 从当前时间到下一个决策点
        while self.current_time < end_time:  # 当前时间小于结束时间
            # 若存在待分配订单和空闲机器人，则为机器人分配订单
            self.assign_order_to_robot()

            # 若某个区域同时存在空闲拣货员和待分配拣货员的拣货位，则为拣货员分配拣货位
            for area_id in self.area_ids:
                # 当前区域同时存在空闲拣货员和待分配拣货员的拣货位时
                self.assign_pick_point_to_picker(area_id)

            # 选择下一个离散点时刻
            # 判断是否移动时钟到下一个离散点:若当前离散点[新订单到达（为机器人分配订单），拣货员拣货完成，机器人移动到拣货点，机器人拣完商品，机器人移动到depot_position]
            # 中的最小值大于当前时间，则移动时钟到下一个离散点
            new_order_arrival_time = self.orders_not_arrived[0].arrive_time  # 新订单到达时刻
            pickers_pick_complete_time = [picker.pick_end_time for picker in self.pickers]  # 所有拣货员拣货完成时刻
            robots_pick_complete_time = [robot.item_pick_complete_time for robot in self.robots]  # 所有机器人拣完商品时刻
            robots_move_to_pick_point_time = [robot.move_to_pick_point_time for robot in self.robots]  # 所有机器人移动到拣货点时刻
            robots_move_to_depot_time = [robot.move_to_depot_time for robot in self.robots]  # 所有机器人移动到depot_position时刻
            # 所有离散时刻
            discrete_times = ([new_order_arrival_time] + pickers_pick_complete_time +
                              robots_pick_complete_time + robots_move_to_depot_time + robots_move_to_pick_point_time)
            # 下一个离散点时刻
            next_discrete_time = min([time for time in discrete_times if time > self.current_time])
            # 更新当前时间
            self.current_time = next_discrete_time

            # 更新该离散点各对象的属性
            # 若当前时间等于新订单到达时刻，则将新订单加入到待分配订单列表中
            if self.current_time == new_order_arrival_time:
                order = self.orders_not_arrived.pop(0)
                self.orders_unassigned.append(order)

            # 若当前时间等于机器人移动到拣货点时刻，则更新机器人所在拣货位的机器人队列
            for robot in self.robots:
                if self.current_time == robot.move_to_pick_point_time:
                    pick_point = self.pick_points[robot.item.pick_point_id]
                    pick_point.robot_queue.append(robot)
                    # 若当前时间该拣货点有拣货员，则更新拣货员的拣货完成时间和该机器人的拣完商品时间
                    if pick_point.picker is not None:
                        # 机器人拣完商品时间
                        robot.item_pick_complete_time = pick_point.picker.pick_end_time + robot.item.pick_time
                        # 拣货员拣货完成时间
                        pick_point.picker.pick_end_time = robot.item_pick_complete_time

            # 若当前时间等于拣货员拣货完成时刻，则更新拣货员的状态，重置拣货位的拣货员对象
            for picker in self.pickers:
                if self.current_time == picker.pick_end_time:
                    picker.state = 'idle'  # 更新拣货员的状态
                    pick_point = picker.pick_point  # 拣货员所在拣货位
                    pick_point.picker = None  # 重置拣货位的拣货员对象
                    picker.pick_point = None  # 重置拣货员的拣货位对象

            # 若当前时间等于机器人拣完商品时刻
            # 更新拣货位的机器人队列，更新机器人所属订单拣选的商品列表，更新机器人所属订单未拣选完成的商品列表
            for robot in self.robots:
                if self.current_time == robot.item_pick_complete_time:
                    # 机器人所在拣货位
                    pick_point = self.pick_points[robot.item.pick_point_id]
                    # 从拣货位的机器人队列中移除该机器人
                    pick_point.robot_queue.remove(robot)
                    # 更新机器人所属订单拣选的商品列表
                    robot.order.picked_items.append(robot.item)
                    # 更新机器人所属订单未拣选完成的商品列表
                    robot.order.unpicked_items.remove(robot.item)
                    # 若机器人所有商品未拣货完成, 则更新机器人移动到拣货点的时间，更新机器人待拣货或正在拣货的商品
                    if len(robot.item_pick_order) > 0:
                        # 更新机器人拣选商品对象
                        robot.item = robot.item_pick_order.pop(0)
                        # 计算机器人移动到拣货点的时间
                        shortest_path_length = self.shortest_path_between_pick_points(robot, robot.item)
                        move_time = shortest_path_length / robot.speed
                        robot.move_to_pick_point_time = self.current_time + move_time
                    # 若机器人所有商品拣货完成，则更新机器人移动到depot_position的时间
                    else:
                        robot.item = None
                        # 更新机器人移动到depot_position的时间
                        shortest_path_length = self.shortest_path_between_pick_points(robot, self.depot_object)
                        move_time = shortest_path_length / robot.speed
                        robot.move_to_depot_time = self.current_time + move_time

            # 若当前时间等于机器人移动到depot_position时刻，则更新机器人的状态，重置机器人的订单对象
            for robot in self.robots:
                if self.current_time == robot.move_to_depot_time:
                    robot.state = 'idle'
                    robot.order = None
            pass
        pass

    def assign_order_to_robot(self):
        """若存在待分配订单和空闲机器人，则为机器人分配订单"""
        while len(self.orders_unassigned) > 0 and len(self.idle_robots) > 0:
            # 选择一个空闲机器人
            robot = self.idle_robots.pop(0)
            # 选择一个待分配订单
            order = self.orders_unassigned.pop(0)
            # 为机器人分配订单
            robot.assign_order(order)
            # 机器人待拣选或正在拣选的商品
            robot.item = robot.item_pick_order.pop(0)
            # 计算机器人移动到订单中首个商品的最短路径
            shortest_path_length = self.shortest_path_between_pick_points(robot, robot.item)
            # 计算机器人移动时间
            move_time = shortest_path_length / robot.speed
            # 更新机器人的状态
            robot.state = 'busy'
            # 更新机器人的工作时间
            robot.working_time += move_time
            # 机器人移动到拣货位时间
            robot.move_to_pick_point_time = self.current_time + move_time

    def assign_pick_point_to_picker(self, area_id):
        """当前区域同时存在空闲拣货员和待分配拣货员的拣货位时"""
        while len(self.idle_pickers[area_id]) > 0 and len(self.idle_pick_points[area_id]) > 0:
            # 随机选择一个空闲拣货员
            picker = random.choice(self.idle_pickers[area_id])
            # 选择距离该拣货员位置最近的待分配拣货位
            pick_point = min(self.idle_pick_points[area_id], key=lambda point: self.shortest_path_between_pick_points(picker, point))
            # 为拣货员分配拣货位
            picker.pick_point = pick_point
            # 为拣货位分配拣货员
            pick_point.picker = picker
            # 计算拣货员移动到拣货位的最短路径长度
            shortest_path_length = self.shortest_path_between_pick_points(picker, pick_point)
            # 计算拣货员移动到拣货位的时间
            move_time = shortest_path_length / picker.speed
            # 更新拣货员的状态
            picker.state = 'busy'
            # 更新拣货员的工作时间
            picker.working_time += move_time
            # 拣货员在该拣货位拣货开始时间
            picker.pick_start_time = self.current_time + move_time
            # 更新该拣货位的机器人对象队列中的所有机器人该商品的拣货完成时间，按机器人队列顺序完成拣货
            for n in range(len(pick_point.robot_queue)):
                robot = pick_point.robot_queue[n]
                # 机器人拣货完成时间
                robot.item_pick_complete_time = picker.pick_start_time + (n + 1) * robot.item.pick_time
            # 拣货员在该拣货位拣货结束时间
            picker.pick_end_time = picker.pick_start_time + len(pick_point.robot_queue) * picker.item.pick_time
            # 更新拣货员的位置
            picker.position = pick_point.position

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
