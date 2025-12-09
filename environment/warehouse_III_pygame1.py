"""
智能仓库人机协同仿真环境
包含订单、商品、拣货位、机器人、拣货员等实体定义和核心仿真逻辑。
已修复：Picker类缺少 run_start_time 和 run_end_time 属性的错误。
"""
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import math
import copy
import pygame
import sys
import time


# ================= 1. 配置类 (基于 class_public.py) =================
class Config:
    """配置类：定义算法和环境参数"""

    def __init__(self):
        self.parameters = self._parameter()  # 配置项

    def _parameter(self):
        """算法和环境参数定义"""
        parameters = {
            "warehouse": {
                # 单层货架中储货位数量
                "shelf_capacity": 20,  # N_l
                # 货架层数 (未使用)
                "shelf_levels": 3,
                # 仓库区域数量
                "area_num": 3,  # N_a
                # 仓库每个区域中巷道数量
                "aisle_num": 3,  # N_ai (每个区域的巷道组数量)
                # 储货位的长度
                "shelf_length": 1.0,  # S_l
                # 储货位的宽度
                "shelf_width": 1.0,  # S_w
                # 底部通道/巷道的宽度
                "aisle_width": 2.0,  # S_b (用于通道和巷道宽度)
                # 仓库的出入口处的宽度 (未使用)
                "entrance_width": 2.0,
                # depot_position: 机器人的起始位置
                "depot_position": (18.0, 0.0)
            },
            "robot": {
                "short_term_unit_run_cost": 110 / (3600 * 8),
                "long_term_unit_run_cost": 1000000 / (3600 * 8 * 30 * 8 * 365),
                "robot_speed": 3.0  # m/s
            },
            "picker": {
                "short_term_unit_time_cost": 360 / (3600 * 8),
                "long_term_unit_time_cost": 7000 / (3600 * 8 * 30),
                "picker_speed": 0.75,
                "unit_fire_cost": 0
            },
            "order": {
                "unit_delay_cost": 0.01,
                "pack_time": 20.0,
                "poisson_parameter": (60, 180),
                "due_time_list": [1800, 3600, 7200],
                "order_n_arrival": (1, 10),
                "order_n_items": (10, 30)
            },
            "item": {
                "pick_time": 10.0  # 秒
            },
            "ppo": {
                # ... (PPO parameters omitted for environment core)
            }
        }
        # 确保尺寸是浮点数
        for key in ["shelf_length", "shelf_width", "aisle_width", "entrance_width"]:
            parameters["warehouse"][key] = float(parameters["warehouse"][key])
        return parameters


# ================= 2. 基础实体类 (基于 class_warehouse.py) =================
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
        self.due_time = due_time
        self.unit_delay_cost = self.parameter["unit_delay_cost"]
        self.assigned = False  # 订单是否已分配给机器人

    def total_delay_cost(self, current_time):
        """计算订单延期总成本"""
        final_time = self.complete_time if self.complete_time is not None else current_time
        if self.due_time is None or final_time <= self.due_time:
            return 0
        else:
            return (final_time - self.due_time) * self.unit_delay_cost


# 拣货位类
class PickPoint:
    def __init__(self, point_id, position, area_id, aisle_idx):
        self.point_id = point_id
        self.position = position  # (x, y) 巷道中心点
        self.area_id = area_id
        self.aisle_idx = aisle_idx
        self.robot_queue = []
        self.picker = None
        # 拣货位上的所有未拣选商品（用于调度）
        self.unpicked_items = []

    @property
    def is_idle(self):
        """监测拣货位置是否待分配拣货员"""
        # 如果拣货位上未分配拣货员且机器人队列中有机器人，则返回True
        return len(self.robot_queue) > 0 and self.picker is None


# ================= 3. 动态移动实体基类 (用于动画) =================

class MovingEntity:
    """机器人和拣货员的基类，处理基于路径点的移动"""

    def __init__(self, entity_id, position, speed, color):
        self.id = entity_id
        self.position = list(position)
        self.speed = speed
        self.color = color
        self.state = 'idle'
        self.waypoints = []  # 路径点队列 [(x1,y1), (x2,y2)...]
        self.current_target = None
        self.remove = False
        self.entity_type = 'entity'

    def set_path(self, waypoints):
        """设置新的移动路径"""
        self.waypoints = waypoints
        if self.waypoints:
            self.current_target = self.waypoints.pop(0)
            self.state = 'moving'

    def update_position(self, dt):
        """根据时间片更新位置"""
        if self.state != 'moving' or self.current_target is None:
            return

        # 计算到当前目标的向量
        dx = self.current_target[0] - self.position[0]
        dy = self.current_target[1] - self.position[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        step_dist = self.speed * dt

        if dist <= step_dist:
            # 到达当前目标点
            self.position = list(self.current_target)
            if self.waypoints:
                # 取下一个路径点
                self.current_target = self.waypoints.pop(0)
            else:
                # 路径走完
                self.current_target = None
        else:
            # 移动一步
            self.position[0] += (dx / dist) * step_dist
            self.position[1] += (dy / dist) * step_dist


# 机器人类 (基于 class_warehouse.py 的 Robot 继承 MovingEntity)
class Robot(MovingEntity, Config):
    def __init__(self, robot_id, position, rent_type):
        Config.__init__(self)  # 初始化Config
        self.parameter = self.parameters["robot"]
        speed = self.parameter["robot_speed"]
        super().__init__(robot_id, position, speed, (0, 100, 255))
        self.entity_type = 'robot'
        self.pick_point = None
        self.order = None
        self.item_pick_order = []
        self.pick_point_complete_time = 0
        self.move_to_pick_point_time = 0
        self.move_to_depot_time = 0
        self.working_time = 0

        # 机器人运行时间追踪
        self.run_start_time = None
        self.run_end_time = None

        self.rent = rent_type  # 'long' or 'short'
        self.unit_time_cost = self.parameter[f"{rent_type}_term_unit_run_cost"]
        self.pick_point_selection_rule = 2  # 默认选择距离最近

        # 仓库尺寸参数 (用于距离计算)
        wh_params = self.parameters["warehouse"]
        self.S_b = wh_params["aisle_width"]
        self.S_l = wh_params["shelf_length"]
        self.N_l = wh_params["shelf_capacity"]

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

    def distance_between_pick_points(self, position1, position2):
        """两个拣货位之间的最短路径长度（若不在一个巷道，则需要从上部或下部绕过储货位）"""
        x1, y1 = position1
        x2, y2 = position2
        if abs(x1 - x2) < 1e-3:  # 同一巷道
            return abs(y1 - y2)
        else:
            # Y = S_b/2 是底部通道中心线的 Y 坐标
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            # Y = S_b * 1.5 + N_l * S_l 是顶部通道中心线的 Y 坐标
            y_top_aisle = self.S_b + self.N_l * self.S_l + self.S_b / 2
            path2 = abs(y1 - y_top_aisle) + abs(y2 - y_top_aisle) + abs(x1 - x2)
            return min(path1, path2)

    def total_run_cost(self, current_time):
        """当前时刻机器人总的运行成本"""
        end_time = self.run_end_time if self.run_end_time is not None else current_time
        if self.run_start_time is None: return 0
        run_time = end_time - self.run_start_time
        return run_time * self.unit_time_cost


# 拣货员类 (基于 class_warehouse.py 的 Picker 继承 MovingEntity)
class Picker(MovingEntity, Config):
    def __init__(self, picker_id, area_id, position, rent_type):
        Config.__init__(self)  # 初始化Config
        self.parameter = self.parameters["picker"]
        speed = self.parameter["picker_speed"]
        super().__init__(picker_id, position, speed, (255, 50, 50))
        self.entity_type = 'picker'
        self.area_id = area_id
        self.pick_point = None
        self.unit_fire_cost = self.parameter["unit_fire_cost"]
        self.pick_start_time = 0
        self.pick_end_time = 0
        self.hire_time = time.time()  # 实际环境中应使用仿真时间
        self.fire_time = None

        # 修复: 确保 Picker 也有 run_start_time 和 run_end_time 属性
        self.run_start_time = None
        self.run_end_time = None

        self.rent = rent_type  # 'long' or 'short'
        self.unit_time_cost = self.parameter[f"{rent_type}_term_unit_time_cost"]
        self.pick_point_selection_rule = 2

        # 仓库尺寸参数 (用于距离计算)
        wh_params = self.parameters["warehouse"]
        self.S_b = wh_params["aisle_width"]
        self.S_l = wh_params["shelf_length"]
        self.N_l = wh_params["shelf_capacity"]

    def distance_between_pick_points(self, position1, position2):
        """两个拣货位之间的最短路径长度（若不在一个巷道，则需要从上部或下部绕过储货位）"""
        x1, y1 = position1
        x2, y2 = position2
        if abs(x1 - x2) < 1e-3:
            return abs(y1 - y2)
        else:
            path1 = abs(y1 - self.S_b / 2) + abs(y2 - self.S_b / 2) + abs(x1 - x2)
            y_top_aisle = self.S_b + self.N_l * self.S_l + self.S_b / 2
            path2 = abs(y1 - y_top_aisle) + abs(y2 - y_top_aisle) + abs(x1 - x2)
            return min(path1, path2)

    def total_hire_cost(self, current_time):
        """当前时刻拣货员总的雇佣成本"""
        fire_time = self.fire_time if self.fire_time is not None else current_time
        if self.hire_time is None: return 0
        hire_time = fire_time - self.hire_time
        total_cost = hire_time * self.unit_time_cost
        if self.fire_time is not None:
            total_cost += self.unit_fire_cost
        return total_cost


# ================= 4. 核心环境类 (WarehouseEnv) =================

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.config_obj = Config()
        self.params = self.config_obj.parameters["warehouse"]
        self.item_params = self.config_obj.parameters["item"]
        self.order_params = self.config_obj.parameters["order"]

        # 布局参数解析
        self.N_a = self.params["area_num"]
        self.area_ids = [f"area{i + 1}" for i in range(self.N_a)]
        self.N_ai_per_area = self.params["aisle_num"]
        self.N_l = self.params["shelf_capacity"]

        # 物理尺寸 (基于 class_public.py)
        self.S_w = self.params["shelf_width"]  # 货架宽度 (1.0)
        self.S_a = self.params["aisle_width"]  # 巷道/通道宽度 (2.0)
        self.S_l = self.params["shelf_length"]  # 储货位长度 (1.0)
        self.S_cross = self.params["aisle_width"]  # 顶部/底部通道宽度 (2.0)

        # 坐标边界 (用于寻路和绘图)
        # Y=0 是底部通道上方边缘
        self.y_bottom_aisle = self.S_cross / 2.0  # 底部通道中心线 Y (1.0)
        self.y_picking_start = self.S_cross  # 拣货区开始 Y (2.0)
        self.y_picking_end = self.y_picking_start + self.N_l * self.S_l  # 拣货区结束 Y (2 + 20*1 = 22.0)
        self.y_top_aisle = self.y_picking_end + self.S_cross / 2.0  # 顶部通道中心线 Y (22 + 1 = 23.0)

        self.depot_position = self.params["depot_position"]  # (18.0, 0.0)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 20.0  # 放大比例，提高清晰度
        self.offset_x = 50
        self.offset_y = 50
        self.max_x = 0.0  # 仓库最大X坐标

        # 初始化数据结构
        self.pick_points = {}
        self.pick_points_area = {aid: [] for aid in self.area_ids}
        self.robots = []
        self.pickers = []
        self.pickers_area = {aid: [] for aid in self.area_ids}
        self.orders_queue = []
        self.active_orders = []
        self.completed_orders = []

        self.current_time = 0.0
        self.robot_counter = 0
        self.picker_counter = 0
        self.item_counter = 0

        self._create_warehouse_layout()

    def _create_warehouse_layout(self):
        """生成拣货点坐标和结构，实现区域间的连续巷道。"""
        current_x = 0.0  # 仓库起始X坐标

        total_aisles = self.N_a * self.N_ai_per_area

        # 遍历所有巷道，实现连续布局
        for aisle_local_idx in range(total_aisles):
            aisle_global_idx = aisle_local_idx + 1

            # 确定巷道所属的区域ID
            area_index = aisle_local_idx // self.N_ai_per_area
            area_id = self.area_ids[area_index]

            # 结构：[货架(Sw)] [巷道中心(Sa)] [货架(Sw)]
            # X坐标：巷道中心 X = current_x + Sw + Sa/2
            center_x = current_x + self.S_w + self.S_a / 2.0

            # 遍历货架层数 (即储货位数量 N_l)
            for nl in range(1, self.N_l + 1):
                # Y坐标：底部通道起始 + (层数-0.5)*货位长
                center_y = self.y_picking_start + (nl - 0.5) * self.S_l

                # 拣货位 ID 命名: AisleIdx-Layer
                p_id = f"{aisle_global_idx}-{nl}"
                pp = PickPoint(p_id, (center_x, center_y), area_id, aisle_global_idx)
                self.pick_points[p_id] = pp
                self.pick_points_area[area_id].append(pp)

            # 移动到下一个货架组起始。一个货架组宽度为 2*S_w (两个货架) + S_a (一个巷道)
            # 布局是连续的：[...][S][A][S][A][S][A]...
            current_x += 2 * self.S_w + self.S_a

        # 记录仓库的总宽度
        self.max_x = current_x

    def _calculate_waypoints(self, entity, start_pos, end_pos):
        """
        核心逻辑：计算绕行路径 (曼哈顿避障)
        使用实体自带的距离计算方法来确定绕行路径。
        """
        if not hasattr(entity, 'distance_between_pick_points'):
            return [end_pos]

        x1, y1 = start_pos
        x2, y2 = end_pos

        # 阈值判断：是否在同一垂直线上 (同一巷道)
        if abs(x1 - x2) < 1e-3:
            return [end_pos]

        # 底部通道中心线 Y (S_b/2)
        y_bottom_aisle_center = entity.S_b / 2.0
        # 顶部通道中心线 Y (S_b * 1.5 + N_l * S_l)
        y_top_aisle_center = entity.S_b + entity.N_l * entity.S_l + entity.S_b / 2

        # 计算两种路径的总距离 (与 entity.distance_between_pick_points 逻辑一致)
        dist_bottom = abs(y1 - y_bottom_aisle_center) + abs(x1 - x2) + abs(y2 - y_bottom_aisle_center)
        dist_top = abs(y1 - y_top_aisle_center) + abs(x1 - x2) + abs(y2 - y_top_aisle_center)

        waypoints = []
        if dist_bottom <= dist_top:
            # 走底部通道
            waypoints.append((x1, y_bottom_aisle_center))  # 垂直移动到通道中心
            waypoints.append((x2, y_bottom_aisle_center))  # 水平移动
            waypoints.append((x2, y2))  # 垂直移动到目标
        else:
            # 走顶部通道
            waypoints.append((x1, y_top_aisle_center))  # 垂直移动到通道中心
            waypoints.append((x2, y_top_aisle_center))  # 水平移动
            waypoints.append((x2, y2))  # 垂直移动到目标

        return waypoints

    def _adjust_resources(self, r_delta, p_deltas):
        """调整资源数量"""
        # 机器人调整 (默认为长租 long)
        if r_delta > 0:
            for _ in range(r_delta):
                self.robot_counter += 1
                r = Robot(self.robot_counter, self.depot_position, 'long')
                self.robots.append(r)
        elif r_delta < 0:
            removals = abs(r_delta)
            count = 0
            # 优先移除闲置的机器人
            for r in self.robots:
                if not r.remove and r.state == 'idle':
                    r.remove = True  # 设置移除标记
                    r.run_end_time = self.current_time  # 结束成本计算
                    count += 1
                    if count >= removals: break
            # 立即移除已标记的机器人
            self.robots = [r for r in self.robots if not r.remove]

        # 拣货员调整 (默认为长租 long)
        aisle_group_width = 2 * self.S_w + self.S_a

        for area_id, delta in p_deltas.items():
            area_index = self.area_ids.index(area_id)

            if delta > 0:
                for _ in range(delta):
                    self.picker_counter += 1

                    # 初始位置：计算该区域X轴的中心
                    x_start = area_index * self.N_ai_per_area * aisle_group_width
                    x_end = (area_index + 1) * self.N_ai_per_area * aisle_group_width

                    # 初始位置：取该区域 X 坐标的中心，Y 坐标设为顶部通道中心
                    avg_x = (x_start + x_end) / 2.0
                    init_pos = (avg_x, self.y_top_aisle)

                    p = Picker(self.picker_counter, area_id, init_pos, 'long')
                    p.hire_time = self.current_time  # 记录雇佣时间
                    self.pickers.append(p)
                    self.pickers_area[area_id].append(p)
            elif delta < 0:
                removals = abs(delta)
                count = 0
                # 优先移除闲置的拣货员
                for p in self.pickers_area[area_id]:
                    if not p.remove and p.state == 'idle':
                        p.remove = True
                        p.fire_time = self.current_time  # 记录解聘时间
                        count += 1
                        if count >= removals: break
            # 立即移除已标记的拣货员
            self.pickers = [p for p in self.pickers if not p.remove]
            self.pickers_area[area_id] = [p for p in self.pickers_area[area_id] if not p.remove]

    def _generate_test_orders(self):
        """生成测试订单"""
        all_pids = list(self.pick_points.keys())
        self.orders_queue = []
        for i in range(20):
            # 随机选择 1 到 3 个拣货点
            num_points = random.randint(1, 3)
            selected_pids = random.sample(all_pids, num_points)

            items = []
            for j, pid in enumerate(selected_pids):
                # 每个拣货点随机 1 到 5 个商品
                num_items = random.randint(1, 5)
                pp = self.pick_points[pid]
                for k in range(num_items):
                    self.item_counter += 1
                    items.append(Item(
                        item_id=self.item_counter,
                        bin_id=f"B-{pid}-{k}",
                        position=pp.position,
                        area_id=pp.area_id,
                        pick_point_id=pid
                    ))

            arrive_time = i * 10.0  # 每2分钟一个订单
            due_time = arrive_time + random.choice(self.order_params["due_time_list"])
            self.orders_queue.append(Order(i, items, arrive_time, due_time))

    def _get_nearest_idle_picker(self, pp):
        """获取区域内最近的闲置拣货员"""
        candidates = [p for p in self.pickers_area[pp.area_id] if p.state == 'idle' and not p.remove]
        if not candidates:
            return None

        # 找到最近的拣货员 (使用 Picker 自己的距离计算方法)
        best_picker = min(candidates, key=lambda p: p.distance_between_pick_points(p.position, pp.position))
        return best_picker

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.robots = []
        self.pickers = []
        self.pickers_area = {aid: [] for aid in self.area_ids}
        self.active_orders = []
        self.completed_orders = []
        self.robot_counter = 0
        self.picker_counter = 0
        self.item_counter = 0

        self._generate_test_orders()

        # 初始资源: 5个机器人，每区1个拣货员
        self._adjust_resources(5, {aid: 1 for aid in self.area_ids})
        return None, {}

    def step(self, robot_change, picker_area1_change, picker_area2_change, picker_area3_change):
        """
        执行仿真一个时间段
        Args:
            robot_change: 机器人调整数量 (+/-)
            picker_area1_change: 1区拣货员调整数量
            picker_area2_change: 2区拣货员调整数量
            picker_area3_change: 3区拣货员调整数量
        """
        # 1. 调整资源
        p_deltas = {
            "area1": picker_area1_change,
            "area2": picker_area2_change,
            "area3": picker_area3_change
        }
        self._adjust_resources(robot_change, p_deltas)

        # 2. 运行一段时间 (1秒仿真时间)
        duration = 1.0
        dt = 0.05  # 时间片
        target_time = self.current_time + duration

        total_reward = 0

        while self.current_time < target_time:
            self.current_time += dt

            # --- 订单到达 ---
            while self.orders_queue and self.orders_queue[0].arrive_time <= self.current_time:
                order = self.orders_queue.pop(0)
                self.active_orders.append(order)

            # --- 机器人调度 ---
            idle_robots = [r for r in self.robots if r.state == 'idle' and not r.remove]
            unassigned_orders = [o for o in self.active_orders if not o.assigned]

            while idle_robots and unassigned_orders:
                robot = idle_robots.pop(0)
                order = unassigned_orders.pop(0)
                order.assigned = True
                robot.assign_order(order)
                robot.run_start_time = self.current_time

                # 任务分派：去第一个拣货点
                first_item = robot.item_pick_order[0]
                target_pp = self.pick_points[first_item.pick_point_id]
                robot.pick_point = target_pp

                waypoints = self._calculate_waypoints(robot, robot.position, target_pp.position)
                robot.set_path(waypoints)

            # --- 拣货员调度 ---
            for pp in self.pick_points.values():
                if pp.is_idle:
                    picker = self._get_nearest_idle_picker(pp)
                    if picker:
                        pp.picker = picker
                        picker.pick_point = pp
                        # 拣货员开始工作，标记为忙碌
                        picker.state = 'busy'
                        # 记录拣货员开始工作的时间
                        if picker.run_start_time is None: picker.run_start_time = self.current_time

                        waypoints = self._calculate_waypoints(picker, picker.position, pp.position)
                        picker.set_path(waypoints)

            # --- 实体状态更新 ---

            # 机器人更新
            for r in self.robots:
                if r.remove: continue  # 标记移除的在下次循环直接跳过或在资源调整时被移除

                if r.state == 'moving':
                    r.update_position(dt)
                    if r.current_target is None:  # 路径走完
                        if abs(r.position[1] - self.depot_position[1]) < 1e-3:  # 到达Depot
                            r.state = 'packing'
                            r.finish_work_time = self.current_time + self.order_params["pack_time"]
                        else:  # 到达拣货点
                            r.state = 'waiting_picker'
                            r.pick_point.robot_queue.append(r)

                elif r.state == 'packing':
                    if self.current_time >= r.finish_work_time:
                        # 订单完成
                        r.state = 'idle'
                        r.order.complete_time = self.current_time
                        total_reward -= r.order.total_delay_cost(self.current_time)  # 扣除延期成本
                        self.active_orders.remove(r.order)
                        self.completed_orders.append(r.order)
                        r.order = None
                        r.run_end_time = self.current_time
                        if r.remove:
                            r.remove = False  # 状态重置，实际在资源调整时被移除

            # 拣货员更新
            for p in self.pickers:
                if p.remove: continue

                if p.state == 'moving':
                    p.update_position(dt)
                    if p.current_target is None:  # 到达拣货点

                        current_pp = p.pick_point
                        if current_pp and current_pp.robot_queue:
                            target_robot = current_pp.robot_queue[0]
                            target_robot.state = 'picking'  # 机器人进入拣货状态

                            # 找出机器人需要在该点拣选的所有商品
                            items_to_pick = [i for i in target_robot.item_pick_order if
                                             i.pick_point_id == p.pick_point.point_id]

                            work_time = sum(i.pick_time for i in items_to_pick)
                            p.pick_end_time = self.current_time + work_time
                            p.state = 'picking'

                        else:
                            # 异常情况：机器人队列为空，拣货员空跑
                            p.state = 'idle'
                            if p.pick_point: p.pick_point.picker = None
                            p.pick_point = None
                            # 记录拣货员空闲结束工作时间
                            p.run_end_time = self.current_time

                elif p.state == 'picking':
                    if self.current_time >= p.pick_end_time:
                        # 拣货任务完成
                        current_pp = p.pick_point

                        if current_pp and current_pp.robot_queue:
                            robot = current_pp.robot_queue.pop(0)
                            robot.state = 'idle'  # 机器人拣货完成

                            # 移除机器人商品列表中的已拣选商品
                            items_picked = [i for i in robot.item_pick_order if i.pick_point_id == current_pp.point_id]
                            robot.item_pick_order = [i for i in robot.item_pick_order if i not in items_picked]

                            if not robot.item_pick_order:
                                # 订单完成，回Depot
                                robot.state = 'moving'
                                waypoints = self._calculate_waypoints(robot, robot.position, self.depot_position)
                                robot.set_path(waypoints)
                            else:
                                # 去下一个点 (简化为取第一个剩余商品所属拣货点)
                                robot.state = 'moving'
                                next_item = robot.item_pick_order[0]
                                next_pp = self.pick_points[next_item.pick_point_id]
                                robot.pick_point = next_pp
                                waypoints = self._calculate_waypoints(robot, robot.position, next_pp.position)
                                robot.set_path(waypoints)

                        # 释放拣货员
                        p.state = 'idle'
                        # 记录拣货员完成工作时间
                        p.run_end_time = self.current_time
                        if current_pp: current_pp.picker = None
                        p.pick_point = None

            # --- 累积成本 ---
            for r in self.robots:
                # 机器人在运行期间产生运行成本 (只要未被移除，就会持续产生)
                if r.run_start_time is not None:
                    total_reward -= r.unit_time_cost * dt
            for p in self.pickers:
                # 拣货员被雇佣期间产生雇佣成本 (只要未被解雇，就会持续产生)
                if p.hire_time is not None:
                    total_reward -= p.unit_time_cost * dt

            # --- 渲染 ---
            if self.render_mode == "human":
                self._render_frame()

        return None, total_reward, False, False, {}

    def _render_frame(self):
        """Pygame 渲染逻辑 (增加清晰度)"""
        if self.screen is None:
            pygame.init()
            # 动态计算窗口大小: 使用 self.max_x
            max_x = self.max_x

            w_px = int(max_x * self.scale) + self.offset_x * 2
            # 顶部和底部通道高度 + 货架高度 + 偏移量
            h_px = int(self.y_top_aisle * self.scale) + self.offset_y * 3

            self.screen = pygame.display.set_mode((w_px, h_px))
            pygame.display.set_caption("智能仓库人机协同仿真")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("SimHei", 18)  # 使用中文字体

        self.screen.fill((245, 245, 245))  # 浅灰背景

        def to_screen(pos):
            """将世界坐标转换为屏幕坐标"""
            return int(pos[0] * self.scale) + self.offset_x, int(pos[1] * self.scale) + self.offset_y

        # 1. 绘制通道区域 (Top & Bottom)
        # 底部通道 (从 Y=0 到 Y=S_cross)
        y_bottom_start = to_screen((0, 0))[1]
        y_bottom_height = int(self.S_cross * self.scale)
        pygame.draw.rect(self.screen, (230, 230, 230),
                         (0, y_bottom_start, self.screen.get_width(), y_bottom_height))
        # 顶部通道 (从 Y=y_picking_end 到 Y=y_picking_end + S_cross)
        y_top_start = to_screen((0, self.y_picking_end))[1]
        y_top_height = int(self.S_cross * self.scale)
        pygame.draw.rect(self.screen, (230, 230, 230),
                         (0, y_top_start, self.screen.get_width(), y_top_height))

        # 2. 绘制货架和拣货点
        shelf_w_px = int(self.S_w * self.scale)
        shelf_l_px = int(self.S_l * self.scale)
        aisle_w_px = int(self.S_a * self.scale)

        for pp in self.pick_points.values():
            px, py = to_screen(pp.position)

            # 左侧货架
            # 货架的宽度是 S_w, 巷道中心是 px，所以左侧货架起点在 px - S_a/2 - S_w
            left_rect = pygame.Rect(px - aisle_w_px / 2 - shelf_w_px, py - shelf_l_px / 2, shelf_w_px, shelf_l_px)
            pygame.draw.rect(self.screen, (100, 149, 237), left_rect)
            pygame.draw.rect(self.screen, (50, 50, 100), left_rect, 1)

            # 右侧货架
            # 右侧货架起点在 px + S_a/2
            right_rect = pygame.Rect(px + aisle_w_px / 2, py - shelf_l_px / 2, shelf_w_px, shelf_l_px)
            pygame.draw.rect(self.screen, (100, 149, 237), right_rect)
            pygame.draw.rect(self.screen, (50, 50, 100), right_rect, 1)

            # 拣货点地面标记 (巷道中心)
            pygame.draw.circle(self.screen, (200, 200, 200), (px, py), 3)

            # 显示排队数
            if len(pp.robot_queue) > 0:
                txt = self.font.render(str(len(pp.robot_queue)), True, (0, 0, 0))
                self.screen.blit(txt, (px - 5, py - 10))

        # 3. 绘制 Depot
        dx, dy = to_screen(self.depot_position)
        pygame.draw.rect(self.screen, (100, 200, 100), (dx - 20, dy - 10, 40, 20))
        depot_txt = self.font.render("集货区", True, (0, 50, 0))
        self.screen.blit(depot_txt, (dx - 15, dy + 10))

        # 4. 绘制机器人
        for r in self.robots:
            rx, ry = to_screen(r.position)
            color = (0, 0, 200) if r.state != 'idle' else (100, 100, 255)
            # 绘制主体 (圆)
            pygame.draw.circle(self.screen, color, (rx, ry), 8)
            # 绘制ID
            id_txt = self.font.render(str(r.id), True, (255, 255, 255))
            self.screen.blit(id_txt, (rx - 4, ry - 8))
            # 绘制目标连线
            if r.state == 'moving' and r.current_target:
                tx, ty = to_screen(r.current_target)
                pygame.draw.line(self.screen, (0, 0, 200), (rx, ry), (tx, ty), 1)

        # 5. 绘制拣货员
        for p in self.pickers:
            px, py = to_screen(p.position)
            color = (200, 0, 0) if p.state != 'idle' else (255, 100, 100)
            # 绘制主体 (方块)
            pygame.draw.rect(self.screen, color, (px - 6, py - 6, 12, 12))

        # 6. 状态面板
        r_count = len([r for r in self.robots])
        p_count = len([p for p in self.pickers])
        o_active = len(self.active_orders)
        o_done = len(self.completed_orders)

        info = f"仿真时间: {int(self.current_time)}s | 机器人数量: {r_count} | 拣货员数量: {p_count} | 待处理订单: {o_active} | 已完成订单: {o_done}"
        info_surf = self.font.render(info, True, (0, 0, 0))
        self.screen.blit(info_surf, (10, 10))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ================= 5. 运行入口 =================

if __name__ == "__main__":
    try:
        # 初始化环境并设置渲染模式
        env = WarehouseEnv(render_mode="human")
        env.reset()

        print("仿真开始... 蓝色圆点:机器人, 红色方块:拣货员, 蓝色矩形:货架")
        print("区域之间现在是连续的巷道，用粗灰线标记概念上的区域边界。")

        # 运行多个决策周期
        for step_i in range(1000):
            # 模拟策略：在第0步增加资源，后续保持不变
            robot_adj = 0
            p1_adj, p2_adj, p3_adj = 0, 0, 0

            if step_i == 0:
                robot_adj = 2  # 增加2个机器人 (总共 5+2=7 个)
                p1_adj, p2_adj, p3_adj = 0, 0, 0  # 每区原有1个
            elif step_i == 10:
                robot_adj = 2
                p1_adj, p2_adj, p3_adj = 3, 3, 3  # 每区增加1个拣货员
            elif step_i == 20:
                # 尝试移除闲置资源
                robot_adj = -2  # 移除2个闲置机器人
                p1_adj, p2_adj, p3_adj = -1, -1, -1  # 移除每区1个闲置拣货员

            print(f"Step {step_i + 1}: 资源调整 -> 机器人:{robot_adj}, 拣货员:[A1:{p1_adj}, A2:{p2_adj}, A3:{p3_adj}]")

            # 调用 step 函数，传递4个资源调整参数
            obs, reward, terminated, truncated, info = env.step(robot_adj, p1_adj, p2_adj, p3_adj)

            if terminated or truncated:
                print("仿真结束。")
                break

    except KeyboardInterrupt:
        print("\n用户中断仿真。")
    except Exception as e:
        print(f"\n仿真过程中发生错误: {e}")
    finally:
        env.close()