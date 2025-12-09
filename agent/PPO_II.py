"""
PPO agent：Proximal Policy Optimization (Optimized)
场景：长租模式 (Long-Term Rental)
优化：
1. 累计多回合数据后更新 (Batch Accumulation)，解决单步数据标准差计算报错问题
2. 安全的优势归一化 (Safe Advantage Normalization)
3. 提升初始随机探索能力
"""
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

# 导入环境 (请确保路径正确)
from environment.warehouse_test2 import WarehouseEnv
from environment.class_public import Config

# 设置可视化
viz = Visdom(env='PPO_Optimized')
viz.line([0], [0], win='ppo2', opts=dict(title='长租模式(Optimized)', xlabel='Episode', ylabel='Cost'))

# -----------------初始化仓库环境---------------------
warehouse = WarehouseEnv()
N_w = warehouse.N_w  # 仓库宽度
N_l = warehouse.N_l  # 仓库长度
N_a = warehouse.N_a  # 仓库区域数量

# ==========================================
# 1. 工具函数与缓冲区 (Utils & Buffer)
# ==========================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化：提升训练启动速度和稳定性"""
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
# 2. 神经网络定义 (Networks)
# ==========================================

class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=N_w, input_width=N_l,
                 scalar_dim=N_a + 1, hidden_dim=128, output_dim=N_a + 1,
                 out_feature_dim=3, fc_layers=10,
                 attn_heads=4, attn_dim=128):
        super(PolicyNetwork, self).__init__()

        # CNN 特征提取 (正交初始化)
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
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        # 视觉特征全连接
        self.visual_fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, out_feature_dim)),
            nn.ReLU()
        )

        # 骨干网络 (Backbone)
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

        # 输出层 (均值)
        # std=0.01 使初始动作均值接近0，避免初始偏差过大
        self.action_mean = layer_init(nn.Linear(hidden_dim, output_dim), std=0.01)

        # *** 关键优化：提高初始标准差 ***
        # 初始化为 0.5 (即 exp(0.5) ≈ 1.65)，大幅增加初始探索范围
        self.action_log_std = nn.Parameter(torch.ones(output_dim) * 1.5)

    def forward(self, matrix_inputs, scalar_inputs):
        cnn_feat = self.cnn(matrix_inputs)
        vis_feat = self.visual_fc(cnn_feat)

        combined = torch.cat([vis_feat, scalar_inputs], dim=1)
        x = self.fc_backbone(combined)

        # Self-Attention
        attn_input = x.unsqueeze(1)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = x + attn_out.squeeze(1) # Residual
        x = self.attn_norm(x)

        mean = self.action_mean(x)
        std = self.action_log_std.expand_as(mean).exp()

        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=N_w, input_width=N_l,
                 scalar_dim=N_a + 1, hidden_dim=128, out_feature_dim=3):
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
# 3. PPO 代理 (Agent)
# ==========================================

class PPOAgent(Config):
    def __init__(self, policy_network, value_network):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = policy_network.to(self.device)
        self.value_network = value_network.to(self.device)

        # 这里的 policy_old 用于采样
        self.policy_old = copy.deepcopy(policy_network).to(self.device)
        self.policy_old.eval()

        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.parameters["ppo"]["learning_rate"]},
            {'params': self.value_network.parameters(), 'lr': self.parameters["ppo"]["learning_rate"]}
        ])

        # 超参数
        self.gamma = self.parameters["ppo"]["gamma"]
        self.eps_clip = self.parameters["ppo"]["clip_range"]
        self.K_epochs = self.parameters["ppo"]["n_epochs"]

        # *** 关键优化：熵系数 ***
        self.ent_coef = self.parameters['ppo'].get('initial_entropy_coeff', 0.05)
        self.ent_coef_decay = self.parameters['ppo'].get('entropy_coeff_decay', 0.995)
        self.min_ent_coef = self.parameters['ppo'].get('min_entropy_coeff', 0.01)

        # Mini-batch 大小
        self.batch_size = 128

        self.mse_loss = nn.MSELoss()
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        """选择动作，不计算梯度"""
        self.policy_old.eval()
        with torch.no_grad():
            # 1. 状态预处理
            matrix_inputs = torch.FloatTensor(np.array([
                state['robot_queue_list'],
                state['picker_list'],
                state['unpicked_items_list']
            ])).unsqueeze(0).to(self.device)

            scalar_inputs = torch.FloatTensor(np.array(
                [state['n_robots']] + state['n_pickers_area']
            )).unsqueeze(0).to(self.device)

            # 2. 网络前向传播
            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)

            # 3. 采样
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1) # Sum log prob over action dimension

            # 4. 获取当前状态价值 (Critic)
            value = self.value_network(matrix_inputs, scalar_inputs)

        # 5. 存入 Buffer
        # 注意：Reward 和 Done 需要在 env.step 后调用 buffer.add 补充
        # 这里我们暂时返回 numpy 数据给外部循环处理
        action_np = action.cpu().numpy().flatten()
        log_prob_np = log_prob.cpu().item()
        value_np = value.cpu().item()

        return action_np, log_prob_np, value_np, matrix_inputs, scalar_inputs

    def update(self):
        """PPO 核心更新逻辑"""
        # 0. 安全检查：如果没有足够的数据（至少2条以计算std），则跳过更新
        if len(self.buffer.rewards) <= 1:
            # 长租模式下我们应该累积数据，如果不小心触发了这里，先保留数据或清空看策略
            # 现在的策略是如果调用了update但数据不够，说明累积还没完成，但函数被强制调用了
            # 为安全起见，如果外部逻辑正确，这里不应经常触发
            return

        # 1. 转换 Buffer 数据为 Tensor
        matrix_states = torch.cat(self.buffer.matrix_states).to(self.device) # [Batch, C, H, W]
        scalar_states = torch.cat(self.buffer.scalar_states).to(self.device) # [Batch, Dim]
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(self.buffer.dones), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(self.buffer.values), dtype=torch.float32).to(self.device).squeeze()

        if values.dim() == 0: values = values.unsqueeze(0)

        # 2. 计算 Monte Carlo 收益 (Returns) 和 优势 (Advantages)
        # 推荐使用 GAE (Generalized Advantage Estimation)
        returns = []
        advantages = []
        gae = 0
        # 下一个状态的价值，对于最后一步假设为0 (如果Done) 或者需要在外部计算next_value
        # 这里为了简化，假设最后一步后无价值，或者 buffer 里存了完整的 episode
        next_value = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - dones[step]) * gae # lam=0.95

            advantages.insert(0, gae)
            # Return = Advantage + Value
            returns.insert(0, gae + values[step])

            next_value = values[step]

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # *** 优化关键点: 优势归一化 (安全修复版) ***
        # 只有当样本数大于1且标准差有效时，才执行除法归一化
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # 如果只有一个样本，归一化没有意义（去均值后为0），或者直接保持原值
            advantages = (advantages - advantages.mean())

        # 3. Mini-batch 训练
        dataset_size = len(rewards)
        batch_size = min(self.batch_size, dataset_size)

        for _ in range(self.K_epochs):
            # 随机采样器
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False)

            for indices in sampler:
                indices = torch.tensor(indices).long()

                # 取出当前 Batch 的数据
                b_matrix = matrix_states[indices]
                b_scalar = scalar_states[indices]
                b_actions = actions[indices]
                b_old_logprobs = old_logprobs[indices]
                b_returns = returns[indices]
                b_advantages = advantages[indices]

                # 重新评估动作概率和价值
                mean, std = self.policy(b_matrix, b_scalar)
                dist = Normal(mean, std)

                logprobs = dist.log_prob(b_actions).sum(dim=1)
                dist_entropy = dist.entropy().sum(dim=1) # 计算熵

                # 修复：确保 value 是 1D Tensor，匹配 b_returns 形状
                state_values = self.value_network(b_matrix, b_scalar).squeeze()
                if state_values.dim() == 0: state_values = state_values.unsqueeze(0)
                b_returns = b_returns.view(-1)

                # 计算 Ratio
                ratios = torch.exp(logprobs - b_old_logprobs)

                # Surrogate Loss
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages

                # *** 探索关键点 2: 加入 Entropy Loss ***
                # loss = -min(surr) + 0.5 * value_loss - 0.01 * entropy
                loss = -torch.min(surr1, surr2).mean() + \
                       0.5 * self.mse_loss(state_values, b_returns) - \
                       self.ent_coef * dist_entropy.mean()

                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪 (防止梯度爆炸)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.optimizer.step()

        # 更新旧的策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 衰减熵系数
        self.ent_coef = max(self.ent_coef * self.ent_coef_decay, self.min_ent_coef)

        # 清空 Buffer
        self.buffer.clear()

# ==========================================
# 4. 数据保存逻辑 (严格保持完整性)
# ==========================================

def get_order_object(order_n_items):
    file_order = 'D:/Python project/DRL_Warehouse/data/instances'
    path = os.path.join(file_order, "orders_{}.pkl".format(order_n_items))
    if not os.path.exists(path):
        print(f"Warning: Order file not found at {path}")
        return []
    with open(path, "rb") as f:
        orders = pickle.load(f)
    return orders

def create_csv_files():
    """创建所有需要的CSV文件并写入表头"""
    # 主CSV文件
    csv_files = {
        'all': 'instance_data_PPO_II_all.csv',
        '2': 'instance_data_PPO_II_2.csv',
        '4': 'instance_data_PPO_II_4.csv',
        '6': 'instance_data_PPO_II_6.csv',
        '10': 'instance_data_PPO_II_10.csv',
        'monthly_config': 'monthly_configurations_PPO_II.csv',
    }

    # 主结果文件表头
    result_header = [
        'Episode', 'Total_Cost', 'Delay_Cost', 'Robot_Cost', 'Picker_Cost',
        'Completed_Orders', 'OnTime_Completed', 'Total_Orders',
        'Avg_Picking_Time', 'Completion_Rate', 'Scenario'
    ]

    # 每月配置方案文件表头
    config_header = [
        'Episode', 'Month', 'Robot_Total',
        'Picker_Area1', 'Picker_Area2', 'Picker_Area3',
        'Monthly_Robot_Cost', 'Monthly_Picker_Cost', 'Monthly_Config_Cost', 'Scenario'
    ]

    # 创建结果文件
    for key in ['all', '2', '4', '6', '10']:
        filename = csv_files[key]
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_header)

    # 创建每月配置方案文件
    config_filename = csv_files['monthly_config']
    if not os.path.exists(config_filename):
        with open(config_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(config_header)

    return csv_files

def save_to_csv(filename, row_data):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

def save_monthly_config_to_csv(filename, episode, month, robot_total, picker_config,
                               monthly_robot_cost, monthly_picker_cost, scenario):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        monthly_config_cost = monthly_robot_cost + monthly_picker_cost
        row = [episode, month, robot_total]
        row.extend(picker_config)
        row.extend([monthly_robot_cost, monthly_picker_cost, monthly_config_cost, scenario])
        writer.writerow(row)

# ==========================================
# 5. 训练主循环
# ==========================================

def train_ppo_agent(ppo_agent, warehouse, num_episodes=1000):
    scenarios = [2, 4, 6, 10]
    csv_files = create_csv_files()
    total_cost_min = float('inf')

    # 成本参数
    robot_long_unit_cost = ppo_agent.parameters['robot']['long_term_unit_run_cost']
    picker_long_unit_cost = ppo_agent.parameters['picker']['long_term_unit_time_cost']
    monthly_seconds = (8 * 3600) * 30

    for episode in range(num_episodes):
        scenario_idx = episode % len(scenarios)
        current_scenario = scenarios[scenario_idx]
        current_scenario = 10 # 强制测试，按需修改

        print(f"Episode {episode + 1}/{num_episodes} - Using scenario: {current_scenario} items")

        orders = get_order_object(current_scenario)
        env = copy.deepcopy(warehouse)
        state = env.reset(orders)

        done = False
        ep_total_reward = 0

        # 长租模式逻辑：每Episode只决策一次 (max_months=1)
        month_count = 0
        max_months_per_episode = 1

        # === 关键修改：不要在Episode开始时清空Buffer ===
        # ppo_agent.buffer.clear()  <-- 删除这行，允许数据跨Episode累积

        while not done and month_count < max_months_per_episode:
            # 1. 选择动作
            action, log_prob, value, mat_in, scal_in = ppo_agent.select_action(state)

            # 2. 环境交互
            next_state, reward, done = env.step(action, first_step=True, pattern='long')
            ep_total_reward += reward

            # 3. 存入 Buffer
            ppo_agent.buffer.add(mat_in, scal_in, action, log_prob, reward, done, value)

            state = next_state

            # --- 数据统计与保存 (完全保留) ---
            current_robot_total = len([robot for robot in env.robots if not robot.remove])
            current_picker_config = [len(env.pickers_area[area_id]) for area_id in env.area_ids]

            monthly_robot_cost = current_robot_total * robot_long_unit_cost * monthly_seconds
            monthly_picker_cost = sum(current_picker_config) * picker_long_unit_cost * monthly_seconds

            save_monthly_config_to_csv(
                csv_files['monthly_config'],
                episode + 1,
                month_count,
                current_robot_total,
                current_picker_config,
                monthly_robot_cost,
                monthly_picker_cost,
                current_scenario
            )

            month_count += 1
            print(f"  Month {month_count}: Robots={current_robot_total}, Pickers={current_picker_config}")

        if month_count >= max_months_per_episode:
            done = True
            print(f"Episode {episode + 1} reached maximum months limit.")

        # 4. 模型更新策略调整 (Batch Accumulation)
        # 只有当Buffer累积了足够的数据（>= batch_size）才更新，避免单样本std报错
        if len(ppo_agent.buffer.rewards) >= ppo_agent.batch_size:
            print(f"Updating agent with {len(ppo_agent.buffer.rewards)} samples...")
            ppo_agent.update()
            # update()内部会清空buffer

        # --- 结果结算与保存 (完全保留) ---
        total_delay_cost = sum([order.total_delay_cost(env.current_time) for order in env.orders_arrived])
        total_robot_cost = sum([robot.total_run_cost(env.current_time) for robot in env.robots_added])
        total_picker_cost = sum([picker.total_hire_cost(env.current_time) for picker in env.pickers_added])
        cost = -ep_total_reward

        completed_orders = env.orders_completed
        completed_count = len(completed_orders)
        total_orders_count = len(env.orders_arrived)
        on_time_count = len([o for o in completed_orders if o.complete_time <= o.due_time])

        avg_picking_time = 0
        if completed_count > 0:
            avg_picking_time = sum([o.complete_time - o.arrive_time for o in completed_orders]) / completed_count

        completion_rate = 0
        if total_orders_count > 0:
            completion_rate = on_time_count / total_orders_count

        print(f"Episode {episode + 1} Result:")
        print(f"  Cost: {cost:.2f} (Delay: {total_delay_cost:.2f}, Robot: {total_robot_cost:.2f}, Picker: {total_picker_cost:.2f})")
        print(f"  Orders: {completed_count}/{total_orders_count}, Rate: {completion_rate:.4f}")
        print("-" * 60)

        row_data = [
            episode + 1, cost, total_delay_cost, total_robot_cost, total_picker_cost,
            completed_count, on_time_count, total_orders_count,
            avg_picking_time, completion_rate, current_scenario
        ]

        # 保存 CSV
        save_to_csv(csv_files['all'], row_data)
        save_to_csv(csv_files[str(current_scenario)], row_data)

        # 可视化
        viz.line([cost], [episode + 1], win='ppo2', update='append')

        # 保存最佳模型
        if cost < total_cost_min:
            torch.save(ppo_agent.policy.state_dict(), f"policy_network_PPO_II.pth")
            total_cost_min = cost

        # 保存简要训练数据
        with open('../result/result_file/training_data_PPO_II.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, ep_total_reward])

if __name__ == "__main__":
    # 初始化环境尺寸
    warehouse = WarehouseEnv()
    N_w = warehouse.N_w
    N_l = warehouse.N_l
    N_a = warehouse.N_a

    # 30天
    total_seconds = (8 * 3600) * 30
    warehouse.total_time = total_seconds

    # 网络初始化
    policy_network = PolicyNetwork(input_height=N_w, input_width=N_l, scalar_dim=N_a+1, output_dim=N_a+1)
    value_network = ValueNetwork(input_height=N_w, input_width=N_l, scalar_dim=N_a+1)

    # Agent 初始化
    ppo_agent = PPOAgent(policy_network, value_network)

    # 开始训练
    train_ppo_agent(ppo_agent, warehouse, num_episodes=3000)