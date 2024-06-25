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
    def __init__(self, order_id, items):
        self.order_id = order_id
        self.items = items  # 订单中的商品列表
        # 订单中的未拣选商品列表
        self.unpicked_items = copy.deepcopy(self.items)
        # 订单中的已拣选商品列表
        self.picked_items = []


class Item:
    def __init__(self, item_id, bin_id, position, area_id):
        self.item_id = item_id  # 商品的编号
        self.bin_id = bin_id  # 商品所在的储货位编号
        self.position = position  # 商品所在的位置
        self.area_id = area_id  # 商品所在的区域编号
        self.pick_time = 0.01  # 拣选时间


# 储货位类
class StorageBin:
    def __init__(self, bin_id, position, area_id, item_id):
        self.bin_id = bin_id  # 储货位的编号
        self.position = position  # 储货位的位置
        self.item_id = item_id  # 储货位中的商品编号
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
    def __init__(self, robot_id, start_position):
        self.robot_id = robot_id  # 机器人的编号
        self.position = start_position  # 机器人的起始位置
        self.current_position = start_position  # 机器人当前位置
        self.current_order = None  # 机器人当前处理的订单
        self.path = []  # 机器人的路径
        self.state = None  # 机器人所处状态：'idle', 'busy', 'moving'
        self.speed = 2  # 机器人移动速度

    def assign_order(self, order):
        self.current_order = order
        self.plan_path()

    def plan_path(self):
        # 订单中的商品拣选顺序规划
        if self.current_order:
            self.path = [item.position for item in self.current_order.items]
        else:
            self.path = []

    def move_to_next_position(self):
        if self.path:
            self.position = self.path.pop(0)


class Picker:
    def __init__(self, picker_id):
        self.picker_id = picker_id  # 拣货员的编号
        self.position = None  # 拣货员当前位置
        self.item = None  # 拣货员当前处理的商品
        self.state = None  # State can be 'idle', 'busy', 'moving'
        self.speed = 1  # 拣货员移动速度
        # 拣货员负责的储货位列表
        self.storage_bins = []
        # 拣货员负责的拣货位列表
        self.pick_points = []

    # 根据负责的拣货位列表中的拣货位的坐标计算拣货员的初始位置（取各拣货位的坐标均值）
    @ property
    def initial_position(self):
        x = np.mean([point.position[0] for point in self.pick_points])
        y = np.mean([point.position[1] for point in self.pick_points])
        position = (x, y)
        return position


class WarehouseEnv:
    def __init__(self, N_l, N_w, S_l, S_w, S_b, S_d, S_a, depot_position):
        self.N_l = N_l  # 单个货架中储货位的数量
        self.N_w = N_w  # 巷道的数量
        self.S_l = S_l  # 储货位的长度
        self.S_w = S_w  # 储货位的宽度
        self.S_b = S_b  # 底部通道的宽度
        self.S_d = S_d  # 仓库的出入口处的宽度
        self.S_a = S_a  # 巷道的宽度
        self.N_w_area = 3  # 仓库中每个区域包含的巷道数量
        self.depot_position = depot_position  # 机器人的起始位置
        self.robots = []  # 机器人列表
        self.pickers = []  # 拣货员列表
        self.pick_points = {}  # 拣货位字典
        self.storage_bins = {}  # 储货位字典
        self.items = {}  # 商品字典

    def create_warehouse_graph(self):
        # 创建仓库图, 包括货架、巷道、储货位和商品
        for nw in range(1, self.N_w + 1):
            # 计算该巷道所处的区域编号
            area_id = math.ceil(nw / self.N_w_area)
            for nl in range(1, self.N_l + 1):
                x = self.S_d + (2 * nw - 1) * self.S_w + (2 * nw - 1) / 2 * self.S_a
                y = self.S_b + (2 * nl - 1) / 2 * self.S_l
                # 计算拣货位的位置
                position = (x, y)

                # 创建该拣货位左侧储货位对象
                bin_id_left = f"{nw}-{nl}-left"
                storage_bin = StorageBin(bin_id_left, position, area_id, None)
                self.storage_bins[bin_id_left] = storage_bin
                # 创建该储货位存储的商品对象
                item_id_left = f"{nw}-{nl}-left-item"
                item = Item(item_id_left, bin_id_left, position, area_id)
                self.items[item_id_left] = item
                # 将商品放入储货位
                storage_bin.item_id = item_id_left

                # 创建该拣货位右侧储货位对象
                bin_id_right = f"{nw}-{nl}-right"
                storage_bin = StorageBin(bin_id_right, position, area_id, None)
                self.storage_bins[bin_id_right] = storage_bin
                # 创建该储货位存储的商品对象
                item_id_right = f"{nw}-{nl}-right-item"
                item = Item(item_id_right, bin_id_right, position, area_id)
                self.items[item_id_right] = item
                # 将商品放入储货位
                storage_bin.item_id = item_id_right

                # 创建拣货位对象
                point_id = f"{nw}-{nl}"
                pick_point = PickPoint(point_id, position, area_id, [item_id_left, item_id_right], [bin_id_left, bin_id_right])
                self.pick_points[point_id] = pick_point

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

    def add_robot(self, robot):
        self.robots.append(robot)

    def add_picker(self, picker):
        self.pickers.append(picker)

    def simulate(self):
        # Implement the main simulation logic here
        pass

    def reset(self):
        # Reset the warehouse to initial state

        pass

    def step(self):
        # Implement one step of the simulation
        pass

    def state_extractor(self):
        # Extract the current state of the warehouse
        pass

    def compute_reward(self):
        # Compute the reward based on the current state of the warehouse
        pass


if __name__ == "__main__":
    # 初始化仓库环境
    warehouse = WarehouseEnv(10, 10, (0, 0))

