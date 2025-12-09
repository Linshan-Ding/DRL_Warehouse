"""
智能仓库人机协同仿真环境 (RL适配版)
1. step() 返回 (obs, reward, done, truncated, info)
2. Robot/Picker 集成 7 种选点策略
3. 包含 Pygame 可视化
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


# ================= 1. 配置与基础数据类 =================

class Config:
    def __init__(self):
        self.parameters = {
            "warehouse": {
                "shelf_capacity": 20,  # N_l: 单排货架储货位数量
                "area_num": 3,  # N_a: 区域数量
                "aisle_num": 3,  # N_ai: 每个区域的巷道数量
                "shelf_length": 1.0,  # S_l: 储货位长度
                "shelf_width": 1.0,  # S_w: 储货位宽度
                "aisle_width": 2.0,  # S_a: 巷道宽度
                "depot_position": (18.0, 0.0)
            },
            "robot": {
                "short_term_unit_run_cost": 110 / (3600 * 8),  # 元/秒
                "long_term_unit_run_cost": 1000000 / (3600 * 8 * 30 * 8 * 365),
                "robot_speed": 3.0  # m/s
            },
            "picker": {
                "short_term_unit_time_cost": 360 / (3600 * 8),  # 元/秒
                "long_term_unit_time_cost": 7000 / (3600 * 8 * 30),
                "picker_speed": 0.75,  # m/s
                "unit_fire_cost": 0
            },
            "order": {
                "unit_delay_cost": 0.01,  # 元/秒
                "pack_time": 20.0,
                "due_time_list": [1800, 3600, 7200],
            },
            "item": {
                "pick_time": 10.0  # 秒
            }
        }


class Item:
    def __init__(self, item_id, pick_point_id, pick_time):
        self.item_id = item_id
        self.pick_point_id = pick_point_id
        self.pick_time = pick_time


class Order:
    def __init__(self, order_id, items, arrive_time, due_time, unit_delay_cost):
        self.order_id = order_id
        self.items = items
        self.arrive_time = arrive_time
        self.due_time = due_time
        self.unit_delay_cost = unit_delay_cost
        self.complete_time = None
        self.assigned = False
        self.picked_items = []
        # 深拷贝商品列表用于逻辑追踪
        self.unpicked_items = list(items)

    def total_delay_cost(self, current_time):
        final_time = self.complete_time if self.complete_time is not None else current_time
        if self.due_time is None or final_time <= self.due_time:
            return 0
        return (final_time - self.due_time) * self.unit_delay_cost


class PickPoint:
    def __init__(self, point_id, position, area_id):
        self.point_id = point_id
        self.position = position
        self.area_id = area_id
        self.robot_queue = []
        self.picker = None
        self.unpicked_items_count = 0  # 统计该点的待拣选任务量

    @property
    def is_idle(self):
        # 有机器人排队且无拣货员
        return len(self.robot_queue) > 0 and self.picker is None


# ================= 2. 移动实体类 (含7种策略) =================

class MovingEntity:
    def __init__(self, entity_id, position, speed):
        self.id = entity_id
        self.position = list(position)
        self.speed = speed
        self.state = 'idle'
        self.waypoints = []
        self.current_target = None
        self.remove = False

        # 布局参数 (用于距离计算)
        self.S_b = 2.0  # 通道宽
        self.N_l = 20  # 货位数量
        self.S_l = 1.0  # 货位长

    def set_path(self, waypoints):
        self.waypoints = list(waypoints)
        if self.waypoints:
            self.current_target = self.waypoints.pop(0)
            self.state = 'moving'

    def update_position(self, dt):
        if self.state != 'moving' or self.current_target is None: return
        dx = self.current_target[0] - self.position[0]
        dy = self.current_target[1] - self.position[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        step = self.speed * dt
        if dist <= step:
            self.position = list(self.current_target)
            if self.waypoints:
                self.current_target = self.waypoints.pop(0)
            else:
                self.current_target = None
        else:
            self.position[0] += (dx / dist) * step
            self.position[1] += (dy / dist) * step

    def distance_to(self, target_pos):
        """曼哈顿避障距离计算"""
        x1, y1 = self.position
        x2, y2 = target_pos
        if abs(x1 - x2) < 0.1: return abs(y1 - y2)  # 同巷道

        y_bot = self.S_b / 2
        y_top = self.S_b + self.N_l * self.S_l + self.S_b / 2
        d_bot = abs(y1 - y_bot) + abs(x1 - x2) + abs(y2 - y_bot)
        d_top = abs(y1 - y_top) + abs(x1 - x2) + abs(y2 - y_top)
        return min(d_bot, d_top)


class Robot(MovingEntity):
    def __init__(self, r_id, position, config, rent_type='long'):
        super().__init__(r_id, position, config["robot"]["robot_speed"])
        self.rent = rent_type
        self.unit_cost = config["robot"][f"{rent_type}_term_unit_run_cost"]
        self.order = None
        self.item_pick_order = []
        self.pick_point = None
        self.run_start_time = None
        self.finish_work_time = 0

        # 策略: 1:MinX, 2:Nearest, 3:MinQueue, 4:MaxQueue, 5:MinItems, 6:MaxItems, 7:Random
        self.pick_point_selection_rule = 2

    def assign_order(self, order):
        self.order = order
        self.item_pick_order = list(order.items)

    def next_pick_point(self, all_pick_points):
        """根据策略选择下一个拣货点"""
        if not self.item_pick_order: return None

        # 候选点: 当前订单剩余商品所在的 PickPoint
        candidate_ids = list(set([item.pick_point_id for item in self.item_pick_order]))
        candidates = [all_pick_points[pid] for pid in candidate_ids if pid in all_pick_points]

        if not candidates: return None
        if len(candidates) == 1: return candidates[0]

        # 策略分支
        if self.pick_point_selection_rule == 1:
            return min(candidates, key=lambda p: p.position[0])
        elif self.pick_point_selection_rule == 2:
            return min(candidates, key=lambda p: self.distance_to(p.position))
        elif self.pick_point_selection_rule == 3:
            return min(candidates, key=lambda p: len(p.robot_queue))
        elif self.pick_point_selection_rule == 4:
            return max(candidates, key=lambda p: len(p.robot_queue))
        elif self.pick_point_selection_rule == 5:
            return min(candidates, key=lambda p: p.unpicked_items_count)
        elif self.pick_point_selection_rule == 6:
            return max(candidates, key=lambda p: p.unpicked_items_count)
        elif self.pick_point_selection_rule == 7:
            return random.choice(candidates)
        return candidates[0]


class Picker(MovingEntity):
    def __init__(self, p_id, area_id, position, config, rent_type='long'):
        super().__init__(p_id, position, config["picker"]["picker_speed"])
        self.area_id = area_id
        self.rent = rent_type
        self.unit_cost = config["picker"][f"{rent_type}_term_unit_time_cost"]
        self.unit_fire_cost = config["picker"]["unit_fire_cost"]
        self.pick_point = None
        self.pick_end_time = 0

        # 成本与时间记录
        self.hire_time = None
        self.fire_time = None
        self.run_start_time = None
        self.run_end_time = None

        # 策略设置
        self.pick_point_selection_rule = 2

    def total_hire_cost(self, current_time):
        end_t = self.fire_time if self.fire_time else current_time
        start_t = self.hire_time if self.hire_time else current_time
        cost = (end_t - start_t) * self.unit_cost
        if self.fire_time: cost += self.unit_fire_cost
        return cost

    def next_pick_point(self, candidates):
        """根据策略选择下一个服务的拣货点 (Candidates为区域内需服务的点)"""
        if not candidates: return None

        if self.pick_point_selection_rule == 1:
            return min(candidates, key=lambda p: p.position[0])
        elif self.pick_point_selection_rule == 2:
            return min(candidates, key=lambda p: self.distance_to(p.position))
        elif self.pick_point_selection_rule == 3:
            return min(candidates, key=lambda p: len(p.robot_queue))
        elif self.pick_point_selection_rule == 4:
            return max(candidates, key=lambda p: len(p.robot_queue))
        elif self.pick_point_selection_rule == 5:
            return min(candidates, key=lambda p: p.unpicked_items_count)
        elif self.pick_point_selection_rule == 6:
            return max(candidates, key=lambda p: p.unpicked_items_count)
        elif self.pick_point_selection_rule == 7:
            return random.choice(candidates)
        return candidates[0]


# ================= 3. 核心环境类 (WarehouseEnv) =================

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.config = Config()
        self.params = self.config.parameters["warehouse"]

        # Action Space: [Robot调整, Area1 Picker调整, Area2 Picker调整, Area3 Picker调整]
        self.action_space = spaces.Box(low=-5, high=5, shape=(4,), dtype=np.float32)

        # Observation Space: 使用Dict方便描述复杂状态
        self.observation_space = spaces.Dict({
            "robot_queue_map": spaces.Box(low=0, high=100, shape=(9, 20), dtype=np.float32),
            "picker_map": spaces.Box(low=0, high=1, shape=(9, 20), dtype=np.float32),
            "n_robots": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            "n_pickers": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32),
            "backlog": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32)
        })

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 20.0
        self.offset = (50, 50)

        # 内部数据容器
        self.pick_points = {}
        self.robots = []
        self.pickers = []
        self.pickers_area = {f"area{i + 1}": [] for i in range(self.params["area_num"])}

        self.orders = []
        self.orders_not_arrived = []
        self.active_orders = []
        self.completed_orders = []
        self.orders_unassigned = []

        self.current_time = 0.0
        self.robot_id_counter = 0
        self.picker_id_counter = 0
        self.max_x = 0

        self._build_layout()

    def _build_layout(self):
        p = self.params
        self.N_w = p["area_num"] * p["aisle_num"]
        self.N_l = p["shelf_capacity"]
        self.area_ids = [f"area{i + 1}" for i in range(p["area_num"])]

        # Y轴布局
        self.y_pick_start = p["aisle_width"]
        self.y_pick_end = self.y_pick_start + p["shelf_capacity"] * p["shelf_length"]
        self.y_top_aisle_center = self.y_pick_end + p["aisle_width"] / 2.0
        self.y_bot_aisle_center = p["aisle_width"] / 2.0

        # X轴布局 & ID映射
        current_x = 0.0
        self.pick_points_matrix = [[None for _ in range(self.N_l)] for _ in range(self.N_w)]

        for nw in range(1, self.N_w + 1):
            area_idx = (nw - 1) // p["aisle_num"]
            area_id = self.area_ids[area_idx]
            center_x = current_x + p["shelf_width"] + p["aisle_width"] / 2.0

            for nl in range(1, self.N_l + 1):
                center_y = self.y_pick_start + (nl - 0.5) * p["shelf_length"]
                pid = f"{nw}-{nl}"
                pp = PickPoint(pid, (center_x, center_y), area_id)
                self.pick_points[pid] = pp
                self.pick_points_matrix[nw - 1][nl - 1] = pp

            current_x += 2 * p["shelf_width"] + p["aisle_width"]
        self.max_x = current_x

    def _get_path(self, entity, target_pos):
        """曼哈顿寻路"""
        x1, y1 = entity.position
        x2, y2 = target_pos
        if abs(x1 - x2) < 0.1: return [target_pos]

        d_bot = abs(y1 - self.y_bot_aisle_center) + abs(x1 - x2) + abs(y2 - self.y_bot_aisle_center)
        d_top = abs(y1 - self.y_top_aisle_center) + abs(x1 - x2) + abs(y2 - self.y_top_aisle_center)

        if d_bot < d_top:
            return [(x1, self.y_bot_aisle_center), (x2, self.y_bot_aisle_center), target_pos]
        else:
            return [(x1, self.y_top_aisle_center), (x2, self.y_top_aisle_center), target_pos]

    def _adjust_resources(self, action):
        r_delta = int(round(action[0]))
        p_deltas = [int(round(x)) for x in action[1:]]

        # 1. Robot 调整
        if r_delta > 0:
            for _ in range(r_delta):
                self.robot_id_counter += 1
                r = Robot(self.robot_id_counter, self.params["depot_position"], self.config.parameters)
                r.run_start_time = self.current_time
                self.robots.append(r)
        elif r_delta < 0:
            count = 0
            sorted_robots = sorted(self.robots, key=lambda r: 0 if r.state == 'idle' else 1)
            for r in sorted_robots:
                if count >= abs(r_delta): break
                if not r.remove:
                    r.remove = True
                    count += 1

        # 2. Picker 调整
        for idx, delta in enumerate(p_deltas):
            if idx >= len(self.area_ids): break
            area_id = self.area_ids[idx]
            if delta > 0:
                for _ in range(delta):
                    self.picker_id_counter += 1
                    area_w = self.max_x / len(self.area_ids)
                    init_x = area_w * idx + area_w / 2
                    init_pos = (init_x, self.y_top_aisle_center)
                    p = Picker(self.picker_id_counter, area_id, init_pos, self.config.parameters)
                    p.hire_time = self.current_time
                    self.pickers.append(p)
                    self.pickers_area[area_id].append(p)
            elif delta < 0:
                count = 0
                sorted_pickers = sorted(self.pickers_area[area_id], key=lambda p: 0 if p.state == 'idle' else 1)
                for p in sorted_pickers:
                    if count >= abs(delta): break
                    if not p.remove:
                        p.remove = True
                        p.fire_time = self.current_time
                        count += 1

        # 清理移除的实体
        self.robots = [r for r in self.robots if not (r.remove and r.state == 'idle')]
        active_pickers = []
        for p in self.pickers:
            if p.remove and p.state == 'idle': continue
            active_pickers.append(p)
        self.pickers = active_pickers
        self.pickers_area = {aid: [p for p in self.pickers if p.area_id == aid] for aid in self.area_ids}

    def _get_obs(self):
        """提取状态用于RL"""
        queue_grid = np.zeros((self.N_w, self.N_l), dtype=np.float32)
        picker_grid = np.zeros((self.N_w, self.N_l), dtype=np.float32)

        for w in range(self.N_w):
            for h in range(self.N_l):
                pp = self.pick_points_matrix[w][h]
                if pp:
                    queue_grid[w, h] = len(pp.robot_queue)
                    picker_grid[w, h] = 1.0 if pp.picker else 0.0

        n_robots = np.array([len(self.robots)], dtype=np.float32)
        n_pickers = np.array([len(self.pickers_area[aid]) for aid in self.area_ids], dtype=np.float32)
        backlog = np.array([len(self.orders_unassigned) + len(self.active_orders)], dtype=np.float32)

        return {
            "robot_queue_map": queue_grid,
            "picker_map": picker_grid,
            "n_robots": n_robots,
            "n_pickers": n_pickers,
            "backlog": backlog
        }

    def reset(self, orders=None, seed=None, options=None):
        super().reset(seed=seed)
        self.robots = []
        self.pickers = []
        self.pickers_area = {aid: [] for aid in self.area_ids}
        self.active_orders = []
        self.completed_orders = []
        self.orders_unassigned = []
        self.current_time = 0.0

        for pp in self.pick_points.values():
            pp.robot_queue = []
            pp.picker = None
            pp.unpicked_items_count = 0

        if orders is not None:
            self.orders = copy.deepcopy(orders)
            self.orders_not_arrived = sorted(self.orders, key=lambda x: x.arrive_time)
            # 初始化热度统计
            for o in self.orders:
                for item in o.items:
                    if item.pick_point_id in self.pick_points:
                        self.pick_points[item.pick_point_id].unpicked_items_count += 1
        else:
            self.orders = []
            self.orders_not_arrived = []

        return self._get_obs(), {}

    def step(self, action):
        """
        核心 Step 函数
        Return: obs, reward, done, truncated, info
        """
        # 1. 执行动作
        self._adjust_resources(action)

        # 2. 仿真推进 (1秒)
        duration = 1.0
        dt = 0.05
        target_time = self.current_time + duration

        while self.current_time < target_time:
            self.current_time += dt

            # A. 订单到达
            while self.orders_not_arrived and self.orders_not_arrived[0].arrive_time <= self.current_time:
                self.orders_unassigned.append(self.orders_not_arrived.pop(0))

            # B. 机器人调度 (应用策略)
            idle_robots = [r for r in self.robots if r.state == 'idle' and not r.remove]
            while idle_robots and self.orders_unassigned:
                robot = idle_robots.pop(0)
                order = self.orders_unassigned.pop(0)
                order.assigned = True
                self.active_orders.append(order)
                robot.assign_order(order)

                # 策略选择目标点
                target_pp = robot.next_pick_point(self.pick_points)
                if target_pp:
                    robot.pick_point = target_pp
                    robot.state = 'moving'
                    robot.set_path(self._get_path(robot, target_pp.position))

            # C. 拣货员调度 (应用策略)
            for aid in self.area_ids:
                idle_pickers = [p for p in self.pickers_area[aid] if p.state == 'idle' and not p.remove]
                if not idle_pickers: continue

                # 候选点: 该区域待服务的点
                candidates = [pp for pp in self.pick_points.values() if pp.area_id == aid and pp.is_idle]
                for picker in idle_pickers:
                    if not candidates: break
                    target_pp = picker.next_pick_point(candidates)
                    if target_pp:
                        picker.pick_point = target_pp
                        target_pp.picker = picker
                        picker.state = 'moving'
                        if picker.run_start_time is None: picker.run_start_time = self.current_time
                        picker.set_path(self._get_path(picker, target_pp.position))
                        candidates.remove(target_pp)

            # D. 实体逻辑更新
            # Robot
            for r in self.robots:
                if r.state == 'moving':
                    r.update_position(dt)
                    if r.current_target is None:
                        if r.pick_point:  # 到拣货位
                            r.state = 'waiting_picker'
                            r.pick_point.robot_queue.append(r)
                            r.pick_point = None
                        else:  # 到Depot
                            r.state = 'packing'
                            r.finish_work_time = self.current_time + self.config.parameters["order"]["pack_time"]
                elif r.state == 'packing':
                    if self.current_time >= r.finish_work_time:
                        if r.order in self.active_orders: self.active_orders.remove(r.order)
                        self.completed_orders.append(r.order)
                        r.order.complete_time = self.current_time
                        r.order = None
                        r.state = 'idle'
            # Picker
            for p in self.pickers:
                if p.state == 'moving':
                    p.update_position(dt)
                    if p.current_target is None:
                        pp = p.pick_point
                        if pp and pp.robot_queue:
                            robot = pp.robot_queue[0]
                            robot.state = 'picking'
                            p.state = 'picking'
                            items_here = [i for i in robot.item_pick_order if i.pick_point_id == pp.point_id]
                            time_cost = sum(i.pick_time for i in items_here)
                            p.pick_end_time = self.current_time + time_cost
                        else:
                            p.state = 'idle'
                            if pp: pp.picker = None
                            p.pick_point = None
                elif p.state == 'picking':
                    if self.current_time >= p.pick_end_time:
                        pp = p.pick_point
                        if pp and pp.robot_queue:
                            robot = pp.robot_queue.pop(0)
                            robot.state = 'idle'
                            picked = [i for i in robot.item_pick_order if i.pick_point_id == pp.point_id]
                            for i in picked:
                                robot.item_pick_order.remove(i)
                                pp.unpicked_items_count = max(0, pp.unpicked_items_count - 1)

                            next_pp = robot.next_pick_point(self.pick_points)
                            robot.state = 'moving'
                            if next_pp:
                                robot.pick_point = next_pp
                                robot.set_path(self._get_path(robot, next_pp.position))
                            else:
                                robot.set_path(self._get_path(robot, self.params["depot_position"]))
                        p.state = 'idle'
                        if pp: pp.picker = None
                        p.pick_point = None
                        p.run_end_time = self.current_time

            if self.render_mode == "human":
                self.render()

        # 3. 计算即时回报
        step_cost = 0
        active_robots = len([r for r in self.robots if r.state != 'idle'])
        step_cost += active_robots * self.config.parameters["robot"]["short_term_unit_run_cost"] * duration
        step_cost += len(self.pickers) * self.config.parameters["picker"]["short_term_unit_time_cost"] * duration
        delay_cost = sum([o.total_delay_cost(self.current_time) for o in self.active_orders])

        reward = -step_cost - delay_cost * 0.001

        # 4. 判断结束
        done = False
        if not self.orders_not_arrived and not self.orders_unassigned and not self.active_orders and self.orders:
            done = True

        obs = self._get_obs()
        info = {
            "time": self.current_time,
            "completed": len(self.completed_orders),
            "cost": step_cost
        }

        return obs, reward, done, False, info

    def render(self):
        if self.screen is None:
            pygame.init()
            w = int(self.max_x * self.scale) + 100
            h = int((self.y_top_aisle_center + 5) * self.scale) + 100
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Warehouse RL Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 12)

        self.screen.fill((255, 255, 255))

        def to_s(pos):
            return int(pos[0] * self.scale) + 50, int(pos[1] * self.scale) + 50

        # Draw Layout
        sw, sl = int(self.params["shelf_width"] * self.scale), int(self.params["shelf_length"] * self.scale)
        aw = int(self.params["aisle_width"] * self.scale)
        for pp in self.pick_points.values():
            px, py = to_s(pp.position)
            pygame.draw.rect(self.screen, (200, 150, 100), (px - aw // 2 - sw, py - sl // 2, sw, sl))  # Left Shelf
            pygame.draw.rect(self.screen, (200, 150, 100), (px + aw // 2, py - sl // 2, sw, sl))  # Right Shelf
            col = (200, 200, 200) if pp.is_idle else (230, 230, 230)
            pygame.draw.circle(self.screen, col, (px, py), 3)
            if pp.robot_queue:
                pygame.draw.circle(self.screen, (255, 0, 0), (px, py), min(10, 2 + len(pp.robot_queue) * 2), 1)

        dx, dy = to_s(self.params["depot_position"])
        pygame.draw.circle(self.screen, (0, 200, 0), (dx, dy), 12)

        for r in self.robots:
            rx, ry = to_s(r.position)
            col = (0, 0, 255) if r.state != 'idle' else (150, 150, 255)
            pygame.draw.circle(self.screen, col, (rx, ry), 6)
        for p in self.pickers:
            px, py = to_s(p.position)
            col = (255, 0, 0) if p.state != 'idle' else (255, 150, 150)
            pygame.draw.rect(self.screen, col, (px - 5, py - 5, 10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen: pygame.quit()


# ================= 6. 测试入口 =================
if __name__ == "__main__":
    # 生成测试订单
    orders = []
    for i in range(20):
        items = []
        for j in range(3):
            pid = f"{random.randint(1, 9)}-{random.randint(1, 20)}"
            items.append(Item(f"i_{i}_{j}", pid, 10.0))
        orders.append(Order(i, items, i * 10, i * 10 + 1800, 0.01))

    # 初始化
    env = WarehouseEnv(render_mode="human")
    obs, info = env.reset(orders=orders)

    print("开始仿真...")

    try:
        steps = 0
        while True:
            # 模拟策略: 前期投入资源
            action = [0, 0, 0, 0]
            if steps == 0:
                action = [4, 1, 1, 1]

            # Step调用，返回标准RL格式
            obs, reward, done, trunc, info = env.step(action)
            steps += 1

            if done:
                print(f"完成! 总耗时: {info['time']:.2f}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()