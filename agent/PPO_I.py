import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import copy
import pickle
import csv
import os
import random
from visdom import Visdom
from torch.utils.data import BatchSampler, SubsetRandomSampler

# 假设这些环境文件在您的项目中存在
from environment.warehouse_test2 import WarehouseEnv
from environment.class_public import Config

# 设置训练数据可视化
viz = Visdom(env='PPO_Optimized')
viz.line([0], [0], win='ppo1', opts=dict(title='短租模式 (Optimized)', xlabel='Episode', ylabel='Cost'))


# ==========================================
# 1. 工具函数与缓冲区 (Utils & Buffer)
# ==========================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化：提升深层强化学习网络的收敛速度和稳定性"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RolloutBuffer:
    """经验回放缓冲区"""

    def __init__(self):
        self.matrix_states = []
        self.scalar_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        del self.matrix_states[:]
        del self.scalar_states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]

    def add(self, matrix_state, scalar_state, action, logprob, reward, done, value):
        self.matrix_states.append(matrix_state)
        self.scalar_states.append(scalar_state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


# ==========================================
# 2. 网络定义 (Networks)
# ==========================================

class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=21, input_width=10,
                 scalar_dim=4, hidden_dim=128, output_dim=4,
                 out_feature_dim=32, attn_heads=4):
        super(PolicyNetwork, self).__init__()

        # CNN 特征提取 (使用正交初始化)
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # 动态计算CNN输出维度
        with torch.no_grad():
            # 假设 input_height 和 input_width 是外部传入的 N_w, N_l
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        # 全连接层处理视觉特征
        self.visual_fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, out_feature_dim)),
            nn.ReLU()
        )

        # 融合层 (视觉 + 标量)
        input_size = out_feature_dim + scalar_dim
        self.fc_backbone = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )

        # 注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 输出层 (Mean)
        # std=0.01 使初始均值接近0
        self.action_mean = layer_init(nn.Linear(hidden_dim, output_dim), std=0.01)

        # *** 探索关键点 1: 提高初始标准差 ***
        # 初始化为 0.5 (即 exp(0.5) ≈ 1.65)，比原来的 0 (std=1.0) 更大，探索更激进
        self.action_log_std = nn.Parameter(torch.ones(output_dim) * 1.5)

    def forward(self, matrix_inputs, scalar_inputs):
        cnn_feat = self.cnn(matrix_inputs)
        vis_feat = self.visual_fc(cnn_feat)

        combined = torch.cat([vis_feat, scalar_inputs], dim=1)
        x = self.fc_backbone(combined)

        # Self-Attention (增加维度适配: [batch, 1, hidden])
        attn_input = x.unsqueeze(1)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = x + attn_out.squeeze(1)  # Residual
        x = self.attn_norm(x)

        mean = self.action_mean(x)
        std = self.action_log_std.expand_as(mean).exp()

        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=21, input_width=10,
                 scalar_dim=4, hidden_dim=128, out_feature_dim=32):
        super(ValueNetwork, self).__init__()

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.visual_fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, out_feature_dim)),
            nn.ReLU()
        )

        input_size = out_feature_dim + scalar_dim
        self.fc_backbone = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )

        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, matrix_inputs, scalar_inputs):
        cnn_feat = self.cnn(matrix_inputs)
        vis_feat = self.visual_fc(cnn_feat)
        combined = torch.cat([vis_feat, scalar_inputs], dim=1)
        x = self.fc_backbone(combined)
        value = self.value_head(x)
        return value


# ==========================================
# 3. PPO Agent (Refactored)
# ==========================================

class PPOAgent(Config):
    def __init__(self, policy_network, value_network):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = policy_network.to(self.device)
        self.value_network = value_network.to(self.device)
        self.policy_old = copy.deepcopy(policy_network).to(self.device)
        self.policy_old.eval()

        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.parameters["ppo"]["learning_rate"]},
            {'params': self.value_network.parameters(), 'lr': self.parameters["ppo"]["learning_rate"]}
        ])

        # 超参数
        self.gamma = self.parameters["ppo"]["gamma"]
        self.eps_clip = self.parameters["ppo"]["clip_range"]
        self.K_epochs = self.parameters["ppo"]["n_epochs"]

        # 熵参数
        self.ent_coef = self.parameters['ppo'].get('initial_entropy_coeff', 0.02)
        self.ent_coef_decay = self.parameters['ppo'].get('entropy_coeff_decay', 0.99)
        self.min_ent_coef = self.parameters['ppo'].get('min_entropy_coeff', 0.001)

        # Mini-batch 参数 (如果config里没有，默认64)
        self.batch_size = 64
        self.mse_loss = nn.MSELoss()

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        """选择动作，不计算梯度"""
        self.policy_old.eval()
        with torch.no_grad():
            matrix_inputs = torch.FloatTensor(np.array([
                state['robot_queue_list'],
                state['picker_list'],
                state['unpicked_items_list']
            ])).unsqueeze(0).to(self.device)

            scalar_inputs = torch.FloatTensor(np.array(
                [state['n_robots']] + state['n_pickers_area']
            )).unsqueeze(0).to(self.device)

            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

            # 获取 Critic Value 用于后续 Buffer
            value = self.value_network(matrix_inputs, scalar_inputs)

            action_np = action.cpu().numpy()[0]
            log_prob_np = log_prob.cpu().numpy()[0]
            value_np = value.cpu().numpy()[0]

            # 返回 raw tensors 用于 buffer 存储，避免 update 时重复转换
            return action_np, log_prob_np, value_np, matrix_inputs, scalar_inputs

    def update(self):
        """PPO 更新逻辑 (Mini-batch + GAE + Entropy)"""
        if len(self.buffer.rewards) <= 1:
            self.buffer.clear()
            return

        # 1. 转换数据为 Tensor
        matrix_states = torch.cat(self.buffer.matrix_states).to(self.device)
        scalar_states = torch.cat(self.buffer.scalar_states).to(self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(self.buffer.dones), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(self.buffer.values), dtype=torch.float32).to(self.device).squeeze()

        # 2. GAE 计算
        returns = []
        advantages = []
        gae = 0
        next_value = 0  # 假设最后一步后价值为0，或可从外部传入

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - dones[step]) * gae  # lam=0.95
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # 优势归一化 (安全处理)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = (advantages - advantages.mean())

        # 3. Mini-batch 训练
        dataset_size = len(rewards)
        batch_size = min(self.batch_size, dataset_size)

        for _ in range(self.K_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False)

            for indices in sampler:
                indices = torch.tensor(indices).long()

                b_matrix = matrix_states[indices]
                b_scalar = scalar_states[indices]
                b_actions = actions[indices]
                b_old_logprobs = old_logprobs[indices]
                b_returns = returns[indices]
                b_advantages = advantages[indices]

                # 新策略评估
                mean, std = self.policy(b_matrix, b_scalar)
                dist = Normal(mean, std)

                logprobs = dist.log_prob(b_actions).sum(dim=1)
                dist_entropy = dist.entropy().sum(dim=1)
                state_values = self.value_network(b_matrix, b_scalar).squeeze()

                # Ratios
                ratios = torch.exp(logprobs - b_old_logprobs)

                # Loss Calculation
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages

                # *** 探索关键点 2: 显式熵奖励 ***
                loss = -torch.min(surr1, surr2).mean() + \
                       0.5 * self.mse_loss(state_values, b_returns) - \
                       self.ent_coef * dist_entropy.mean()

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 衰减熵系数
        self.ent_coef = max(self.ent_coef * self.ent_coef_decay, self.min_ent_coef)
        # 清空 Buffer
        self.buffer.clear()


# ==========================================
# 4. 数据处理与保存逻辑 (Data IO)
# ==========================================

def get_order_object(order_n_items):
    file_order = 'D:/Python project/DRL_Warehouse/data/instances'
    # 请确保路径和文件名正确
    path = os.path.join(file_order, "orders_{}.pkl".format(order_n_items))
    if not os.path.exists(path):
        print(f"Warning: Order file not found at {path}, generating empty list.")
        return []
    with open(path, "rb") as f:
        orders = pickle.load(f)
    return orders


def create_csv_files():
    """创建所有需要的CSV文件并写入表头"""
    csv_files = {
        'all': 'instance_data_PPO_I_all.csv',
        '2': 'instance_data_PPO_I_2.csv',
        '4': 'instance_data_PPO_I_4.csv',
        '6': 'instance_data_PPO_I_6.csv',
        '10': 'instance_data_PPO_I_10.csv',
        'daily_config': 'daily_configurations_PPO_I.csv',
    }

    result_header = [
        'Episode', 'Total_Cost', 'Delay_Cost', 'Robot_Cost', 'Picker_Cost',
        'Completed_Orders', 'OnTime_Completed', 'Total_Orders',
        'Avg_Picking_Time', 'Completion_Rate', 'Scenario'
    ]

    config_header = [
        'Episode', 'Day', 'Robot_Total',
        'Picker_Area1', 'Picker_Area2', 'Picker_Area3',
        'Daily_Robot_Cost', 'Daily_Picker_Cost', 'Daily_Config_Cost', 'Scenario'
    ]

    for key in ['all', '2', '4', '6', '10']:
        filename = csv_files[key]
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_header)

    config_filename = csv_files['daily_config']
    if not os.path.exists(config_filename):
        with open(config_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(config_header)

    return csv_files


def save_to_csv(filename, row_data):
    """通用保存函数"""
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def save_daily_config_to_csv(filename, episode, day, robot_total, picker_config,
                             daily_robot_cost, daily_picker_cost, scenario):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        daily_config_cost = daily_robot_cost + daily_picker_cost
        row = [episode, day, robot_total]
        row.extend(picker_config)
        row.extend([daily_robot_cost, daily_picker_cost, daily_config_cost, scenario])
        writer.writerow(row)


# ==========================================
# 5. 训练主循环 (Training Loop)
# ==========================================

def train_ppo_agent(ppo_agent, warehouse, num_episodes=1000):
    scenarios = [2, 4, 6, 10]
    csv_files = create_csv_files()
    total_cost = float('inf')

    # 成本参数读取
    robot_unit_cost = ppo_agent.parameters['robot']['short_term_unit_run_cost']
    picker_unit_cost = ppo_agent.parameters['picker']['short_term_unit_time_cost']
    daily_seconds = 8 * 3600

    for episode in range(num_episodes):
        scenario_idx = episode % len(scenarios)
        current_scenario = scenarios[scenario_idx]
        current_scenario = 10  # 强制测试场景2，按需修改

        print(f"Episode {episode + 1}/{num_episodes} - Using scenario: {current_scenario} items")

        orders = get_order_object(current_scenario)
        env = copy.deepcopy(warehouse)
        state = env.reset(orders)

        done = False
        ep_total_reward = 0

        # 每日循环控制
        day_count = 0
        max_days_per_episode = 30

        # Episode 开始前清空buffer (保险起见)
        ppo_agent.buffer.clear()

        while not done and day_count < max_days_per_episode:
            # 1. 选择动作
            action, log_prob, value, mat_in, scal_in = ppo_agent.select_action(state)

            # 2. 环境交互
            next_state, reward, done = env.step(action)
            ep_total_reward += reward

            # 3. 存入 Buffer
            ppo_agent.buffer.add(mat_in, scal_in, action, log_prob, reward, done, value)
            state = next_state

            # --- 数据统计与 CSV 保存逻辑 (完全保留) ---
            current_robot_total = len([robot for robot in env.robots if not robot.remove])
            current_picker_config = [len(env.pickers_area[area_id]) for area_id in env.area_ids]

            daily_robot_cost = current_robot_total * robot_unit_cost * daily_seconds
            daily_picker_cost = sum(current_picker_config) * picker_unit_cost * daily_seconds

            save_daily_config_to_csv(
                csv_files['daily_config'],
                episode + 1,
                day_count,
                current_robot_total,
                current_picker_config,
                daily_robot_cost,
                daily_picker_cost,
                current_scenario
            )

            day_count += 1
            if day_count % 5 == 0:
                print(f"  Day {day_count}: Robots={current_robot_total}, Pickers={current_picker_config}")

        # 循环结束判定
        if day_count >= max_days_per_episode:
            done = True
            print(f"Episode {episode + 1} reached maximum days limit ({max_days_per_episode})")

        # 4. 模型更新 (每 Episode 更新一次，内部使用 Mini-batch)
        ppo_agent.update()

        # --- 结果计算与保存 (完全保留) ---
        total_delay_cost = sum([order.total_delay_cost(env.current_time) for order in env.orders_arrived])
        total_robot_cost = sum([robot.total_run_cost(env.current_time) for robot in env.robots_added])
        total_picker_cost = sum([picker.total_hire_cost(env.current_time) for picker in env.pickers_added])
        cost = -ep_total_reward

        completed_orders = env.orders_completed
        completed_count = len(completed_orders)
        total_orders_count = len(env.orders_arrived)
        on_time_completed_count = len([o for o in completed_orders if o.complete_time <= o.due_time])

        avg_picking_time = 0
        if completed_count > 0:
            avg_picking_time = sum([o.complete_time - o.arrive_time for o in completed_orders]) / completed_count

        completion_rate = 0
        if total_orders_count > 0:
            completion_rate = on_time_completed_count / total_orders_count

        print(f"Episode {episode + 1} Result:")
        print(
            f"  Cost: {cost:.2f} (Delay: {total_delay_cost:.2f}, Robot: {total_robot_cost:.2f}, Picker: {total_picker_cost:.2f})")
        print(f"  Orders: {completed_count}/{total_orders_count}, Rate: {completion_rate:.4f}")
        print("-" * 60)

        result_row = [
            episode + 1, cost, total_delay_cost, total_robot_cost, total_picker_cost,
            completed_count, on_time_completed_count, total_orders_count,
            avg_picking_time, completion_rate, current_scenario
        ]

        # 保存到主文件和算例文件
        save_to_csv(csv_files['all'], result_row)
        save_to_csv(csv_files[str(current_scenario)], result_row)

        # 可视化
        viz.line([cost], [episode + 1], win='ppo1', update='append')

        # 保存模型
        if cost < total_cost:  # 假设 cost 越小越好
            torch.save(ppo_agent.policy.state_dict(), f"policy_network_PPO_I.pth")
            total_cost = cost

        # 保存简要训练数据
        with open('../result/result_file/training_data_PPO_I.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, ep_total_reward])


if __name__ == "__main__":
    # 初始化环境
    warehouse = WarehouseEnv()
    N_w = warehouse.N_w
    N_l = warehouse.N_l
    N_a = warehouse.N_a

    # 持续时间
    total_seconds = (8 * 3600) * 30
    warehouse.total_time = total_seconds

    # 初始化网络
    # 注意：input_height 和 input_width 应与环境尺寸匹配
    policy_network = PolicyNetwork(input_height=N_w, input_width=N_l, scalar_dim=N_a + 1, output_dim=N_a + 1)
    value_network = ValueNetwork(input_height=N_w, input_width=N_l, scalar_dim=N_a + 1)

    # 初始化 Agent
    ppo_agent = PPOAgent(policy_network, value_network)

    # 开始训练
    train_ppo_agent(ppo_agent, warehouse, num_episodes=3000)