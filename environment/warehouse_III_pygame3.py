"""
智能仓库人机协同仿真环境 (Final HD Version)
1. 强化学习接口: step() 返回 (obs, reward, done, truncated, info)
2. 策略集成: Robot/Picker 保留 7 种智能选点策略
3. 可视化: 高清晰度 Pygame 绘图 (双侧货架、ID显示、实时面板)
"""

import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import math
import copy
import pygame
import pickle
import sys


# ================= 1. 配置与基础类 =================

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
                "short_term_unit_run_cost": 110 / (3600 * 8),
                "long_term_unit_run_cost": 1000000 / (3600 * 8 * 30 * 8 * 365),
                "robot_speed": 3.0
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
                "due_time_list": [1800, 3600, 7200],
            },
            "item": {
                "pick_time": 10.0
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
        self.unpicked_items_count = 0  # 统计用

    @property
    def is_idle(self):
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

        # 缓存布局参数用于距离计算
        self.S_b = 2.0
        self.N_l = 20
        self.S_l = 1.0

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
        """曼哈顿避障距离"""
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
        if not self.item_pick_order: return None
        candidate_ids = list(set([item.pick_point_id for item in self.item_pick_order]))
        candidates = []
        for pid in candidate_ids:
            if pid in all_pick_points: candidates.append(all_pick_points[pid])

        if not candidates: return None
        if len(candidates) == 1: return candidates[0]

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
        self.hire_time = None
        self.fire_time = None
        self.run_start_time = None
        self.run_end_time = None
        self.pick_point_selection_rule = 2

    def total_hire_cost(self, current_time):
        end_t = self.fire_time if self.fire_time else current_time
        start_t = self.hire_time if self.hire_time else current_time
        cost = (end_t - start_t) * self.unit_cost
        if self.fire_time: cost += self.unit_fire_cost
        return cost

    def next_pick_point(self, candidates):
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


# ================= 3. 核心环境类 =================

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.config = Config()
        self.params = self.config.parameters["warehouse"]

        # Action Space: [Robot, A1, A2, A3]
        self.action_space = spaces.Box(low=-5, high=5, shape=(4,), dtype=np.float32)
        # Observation Space (Dict)
        self.observation_space = spaces.Dict({
            "robot_queue_map": spaces.Box(low=0, high=100, shape=(9, 20), dtype=np.float32),
            "picker_map": spaces.Box(low=0, high=1, shape=(9, 20), dtype=np.float32),
            "n_robots": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            "n_pickers": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        })

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 25.0  # 高清缩放
        self.offset = (60, 80)  # 边距偏移 (留出顶部信息栏)

        self.pick_points = {}
        self.robots = []
        self.pickers = []
        self.pickers_area = {f"area{i + 1}": [] for i in range(self.params["area_num"])}

        self.orders = []
        self.orders_not_arrived = []
        self.orders_unassigned = []
        self.active_orders = []
        self.completed_orders = []

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

        self.y_pick_start = p["aisle_width"]
        self.y_pick_end = self.y_pick_start + p["shelf_capacity"] * p["shelf_length"]
        self.y_top_aisle_center = self.y_pick_end + p["aisle_width"] / 2.0
        self.y_bot_aisle_center = p["aisle_width"] / 2.0

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

        self.robots = [r for r in self.robots if not (r.remove and r.state == 'idle')]
        active_pickers = []
        for p in self.pickers:
            if p.remove and p.state == 'idle': continue
            active_pickers.append(p)
        self.pickers = active_pickers
        self.pickers_area = {aid: [p for p in self.pickers if p.area_id == aid] for aid in self.area_ids}

    def _get_obs(self):
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
        return {"robot_queue_map": queue_grid, "picker_map": picker_grid, "n_robots": n_robots, "n_pickers": n_pickers}

    def reset(self, orders=None, seed=None, options=None):
        super().reset(seed=seed)
        self.robots = []
        self.pickers = []
        self.pickers_area = {aid: [] for aid in self.area_ids}
        self.active_orders = []
        self.completed_orders = []
        self.orders_unassigned = []
        self.current_time = 400.0

        for pp in self.pick_points.values():
            pp.robot_queue = []
            pp.picker = None
            pp.unpicked_items_count = 0

        if orders is not None:
            self.orders = copy.deepcopy(orders)
            self.orders_not_arrived = sorted(self.orders, key=lambda x: x.arrive_time)
            for o in self.orders:
                for item in o.items:
                    if item.pick_point_id in self.pick_points:
                        self.pick_points[item.pick_point_id].unpicked_items_count += 1
        else:
            self.orders = []
            self.orders_not_arrived = []

        return self._get_obs(), {}

    def step(self, action):
        self._adjust_resources(action)
        duration = 1.0
        dt = 0.5
        target_time = self.current_time + duration

        while self.current_time < target_time:
            self.current_time += dt

            # Orders
            while self.orders_not_arrived and self.orders_not_arrived[0].arrive_time <= self.current_time:
                self.orders_unassigned.append(self.orders_not_arrived.pop(0))

            # Robot Dispatch
            idle_robots = [r for r in self.robots if r.state == 'idle' and not r.remove]
            while idle_robots and self.orders_unassigned:
                robot = idle_robots.pop(0)
                order = self.orders_unassigned.pop(0)
                order.assigned = True
                self.active_orders.append(order)
                robot.assign_order(order)
                target_pp = robot.next_pick_point(self.pick_points)
                if target_pp:
                    robot.pick_point = target_pp
                    robot.state = 'moving'
                    robot.set_path(self._get_path(robot, target_pp.position))

            # Picker Dispatch
            for aid in self.area_ids:
                idle_pickers = [p for p in self.pickers_area[aid] if p.state == 'idle' and not p.remove]
                if not idle_pickers: continue
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

            # Entity Update
            for r in self.robots:
                if r.state == 'moving':
                    r.update_position(dt)
                    if r.current_target is None:
                        if r.pick_point:
                            r.state = 'waiting_picker'
                            r.pick_point.robot_queue.append(r)
                            r.pick_point = None
                        else:
                            r.state = 'packing'
                            r.finish_work_time = self.current_time + self.config.parameters["order"]["pack_time"]
                elif r.state == 'packing':
                    if self.current_time >= r.finish_work_time:
                        if r.order in self.active_orders: self.active_orders.remove(r.order)
                        self.completed_orders.append(r.order)
                        r.order.complete_time = self.current_time
                        r.order = None
                        r.state = 'idle'

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

        # Reward Calculation
        step_cost = 0
        active_robots = len([r for r in self.robots if r.state != 'idle'])
        step_cost += active_robots * self.config.parameters["robot"]["short_term_unit_run_cost"] * duration
        step_cost += len(self.pickers) * self.config.parameters["picker"]["short_term_unit_time_cost"] * duration
        delay_cost = sum([o.total_delay_cost(self.current_time) for o in self.active_orders])
        reward = -step_cost - delay_cost * 0.001

        done = False
        if not self.orders_not_arrived and not self.orders_unassigned and not self.active_orders and self.orders:
            done = True

        info = {"current_time": self.current_time, "completed": len(self.completed_orders)}
        return self._get_obs(), reward, done, False, info

    def render(self):
        """高清晰度 Pygame 渲染"""
        if self.screen is None:
            pygame.init()
            w = int(self.max_x * self.scale) + self.offset[0] * 2
            h = int((self.y_top_aisle_center + 6) * self.scale) + self.offset[1] * 2
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("智能仓库人机协同仿真 (HD)")
            self.clock = pygame.time.Clock()
            # 设置字体
            self.font = pygame.font.SysFont("Arial", 14)
            self.font_bold = pygame.font.SysFont("Arial", 18, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 10)

        # 1. 背景与区域
        self.screen.fill((245, 245, 245))

        def to_s(pos):
            return int(pos[0] * self.scale) + self.offset[0], int(pos[1] * self.scale) + self.offset[1]

        # 2. 绘制通道 (Top & Bottom Aisle)
        y_bot_start = to_s((0, 0))[1]
        y_bot_h = int(self.params["aisle_width"] * self.scale)
        pygame.draw.rect(self.screen, (220, 220, 220), (0, y_bot_start, self.screen.get_width(), y_bot_h))

        y_top_start = to_s((0, self.y_pick_end))[1]
        pygame.draw.rect(self.screen, (220, 220, 220), (0, y_top_start, self.screen.get_width(), y_bot_h))

        # 3. 绘制货架与拣货点
        sw, sl = int(self.params["shelf_width"] * self.scale), int(self.params["shelf_length"] * self.scale)
        aw = int(self.params["aisle_width"] * self.scale)

        for pp in self.pick_points.values():
            px, py = to_s(pp.position)

            # 双侧货架: 浅木色 + 边框
            shelf_col = (173, 216, 230)  # BurlyWood
            border_col = (0, 0, 0)

            # 左侧货架
            left_rect = (px - aw // 2 - sw, py - sl // 2, sw, sl)
            pygame.draw.rect(self.screen, shelf_col, left_rect)
            pygame.draw.rect(self.screen, border_col, left_rect, 1)

            # 右侧货架
            right_rect = (px + aw // 2, py - sl // 2, sw, sl)
            pygame.draw.rect(self.screen, shelf_col, right_rect)
            pygame.draw.rect(self.screen, border_col, right_rect, 1)

            # 拣货点节点
            col = (200, 200, 200) if pp.is_idle else (100, 100, 100)
            pygame.draw.circle(self.screen, col, (px, py), 3)

            # 排队数量显示
            if pp.robot_queue:
                pygame.draw.circle(self.screen, (255, 69, 0), (px, py), 10)
                txt = self.font_small.render(str(len(pp.robot_queue)), True, (255, 255, 255))
                self.screen.blit(txt, (px - 3, py - 6))

        # 4. 绘制 Depot
        dx, dy = to_s(self.params["depot_position"])
        pygame.draw.rect(self.screen, (34, 139, 34), (dx - 15, dy - 10, 30, 20))  # ForestGreen
        txt_depot = self.font_small.render("Depot", True, (255, 255, 255))
        self.screen.blit(txt_depot, (dx - 13, dy - 6))

        # 5. 绘制机器人
        for r in self.robots:
            rx, ry = to_s(r.position)
            col = (0, 100, 255) if r.state != 'idle' else (135, 206, 250)
            # 连线
            if r.state == 'moving' and r.current_target:
                tx, ty = to_s(r.current_target)
                pygame.draw.line(self.screen, (100, 100, 100), (rx, ry), (tx, ty), 1)
            # 本体
            pygame.draw.circle(self.screen, col, (rx, ry), 8)
            pygame.draw.circle(self.screen, (0, 0, 100), (rx, ry), 8, 1)
            # ID
            txt_id = self.font_small.render(str(r.id), True, (255, 255, 255))
            self.screen.blit(txt_id, (rx - 3, ry - 6))

        # 6. 绘制拣货员
        for p in self.pickers:
            px, py = to_s(p.position)
            col = (220, 20, 60) if p.state != 'idle' else (255, 160, 122)
            pygame.draw.rect(self.screen, col, (px - 6, py - 6, 12, 12))
            pygame.draw.rect(self.screen, (100, 0, 0), (px - 6, py - 6, 12, 12), 1)

        # 7. 顶部 UI 面板
        pygame.draw.rect(self.screen, (50, 50, 50), (0, 0, self.screen.get_width(), 60))

        info_1 = f"Time: {int(self.current_time)}s  |  Orders Completed: {len(self.completed_orders)} / {len(self.orders) + len(self.completed_orders)}"
        info_2 = f"Active Robots: {len(self.robots)}  |  Active Pickers: {len(self.pickers)}  |  Backlog: {len(self.orders_unassigned)}"

        surf_1 = self.font_bold.render(info_1, True, (255, 255, 255))
        surf_2 = self.font.render(info_2, True, (200, 200, 200))

        self.screen.blit(surf_1, (20, 10))
        self.screen.blit(surf_2, (20, 35))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen: pygame.quit()


# ================= 5. 测试入口 =================
if __name__ == "__main__":
    # # 生成测试订单
    # orders = []
    # for i in range(30):
    #     items = []
    #     for j in range(random.randint(2, 5)):
    #         pid = f"{random.randint(1, 9)}-{random.randint(1, 20)}"
    #         items.append(Item(f"i_{i}_{j}", pid, 10.0))
    #     orders.append(Order(i, items, i * 10, i * 10 + 3600, 0.01))

    # 基于仓库中的商品创建一个月内的订单对象，每个订单包含多个商品，订单到达时间服从泊松分布，仿真周期设置为一个月
    num_items = 2

    # 订单数据读取
    file_order = 'D:\\Python project\\DRL_Warehouse\\data\\instances'
    with open(file_order + "\\orders_{}.pkl".format(num_items), "rb") as f:
        orders = pickle.load(f)  # 读取订单数据

    # 初始化
    env = WarehouseEnv(render_mode="human")
    obs, info = env.reset(orders=orders)

    print("环境已启动 (HD模式)...")

    try:
        steps = 0
        while True:
            # 模拟动作: 初始增加资源，后续保持
            action = [0, 0, 0, 0]
            if steps == 0: action = [5, 2, 2, 2]  # 5个机器人，每区2个拣货员

            obs, reward, done, trunc, info = env.step(action)
            steps += 1

            if done:
                print(f"仿真完成! 总步数: {steps}, 总时间: {info['current_time']:.2f}s")
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()