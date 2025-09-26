"""
PPO agent：Proximal Policy Optimization
调整每个决策点机器人和拣货员的数量：拣货员+长短租结合
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from environment.warehouse_III import WarehouseEnv  # 导入仓库环境类
import numpy as np
import copy
import pickle # 用于读取订单数据
from data.generat_order import GenerateData
from visdom import Visdom
import csv
from environment.class_public import Config
import random

# 设置训练数据可视化
viz = Visdom(env='PPO_III')
viz.line([0], [0], win='reward_III', opts=dict(title='Reward3', xlabel='Episode', ylabel='Reward'))

# -----------------初始化仓库环境---------------------
warehouse = WarehouseEnv()
N_w = warehouse.N_w  # 仓库宽度
N_l = warehouse.N_l  # 仓库长度
N_a = warehouse.N_a # 仓库区域数量

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=N_w, input_width=N_l,
                 scalar_dim=N_a + 1, hidden_dim=128, output_dim=N_a + 1,
                 out_feature_dim=3, fc_layers=10,
                 attn_heads=4, attn_dim=128):
        super(PolicyNetwork, self).__init__()
        self.output_dim = output_dim
        self.fc_layers = fc_layers
        self.attn_heads = attn_heads
        self.attn_dim = attn_dim

        # CNN层（保持原结构）
        self.conv1 = nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 注意力机制相关参数
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            batch_first=True
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.attn_fc = nn.Linear(hidden_dim, attn_dim)

        # 动态构建全连接层（增加维度匹配）
        conv_output_size = (input_height // 2) * (input_width // 2) * 1
        self.fc0 = nn.Linear(conv_output_size, out_feature_dim)

        # 调整输入维度以适应注意力
        input_size = out_feature_dim + scalar_dim

        self.fc_modules = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(input_size)

        # 增强全连接层结构
        for i in range(fc_layers):
            in_features = input_size if i == 0 else hidden_dim
            out_features = hidden_dim if i != fc_layers - 1 else attn_dim
            self.fc_modules.append(nn.Linear(in_features, out_features))
            if i < fc_layers - 1:
                self.fc_modules.append(nn.ReLU())

        # 注意力输出适配层
        self.attn_adapter = nn.Sequential(
            nn.Linear(attn_dim*2, hidden_dim),
            nn.Tanh()
        )

        # 最终输出层
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, matrix_inputs, scalar_inputs):
        # 视觉特征提取
        x = self.conv1(matrix_inputs)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # 基础特征处理
        visual_features = torch.relu(self.fc0(x))
        combined = torch.cat((visual_features, scalar_inputs), dim=1)
        combined = self.layer_norm(combined)

        # 动态全连接处理
        for layer in self.fc_modules:
            combined = layer(combined)

        # 自注意力机制
        batch_size = combined.size(0)
        attn_input = combined.unsqueeze(1)  # [batch, 1, features]

        # 计算注意力
        attn_output, _ = self.self_attn(
            query=attn_input,
            key=attn_input,
            value=attn_input
        )
        attn_output = attn_output.squeeze(1)
        attn_output = self.attn_layer_norm(attn_output)

        # 特征融合
        fused_features = torch.cat([combined, attn_output], dim=1)
        fused_features = self.attn_adapter(fused_features)

        # 最终输出
        mean = self.action_mean(fused_features)
        std = torch.exp(self.action_log_std.expand_as(mean))

        return mean, std

# 定义值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=N_w, input_width=N_l,
                 scalar_dim=N_a + 1, hidden_dim=128, out_feature_dim=3,
                 num_fc_layers=10):  # 新增参数控制全连接层数量
        super(ValueNetwork, self).__init__()

        # CNN层
        self.conv1 = nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入尺寸
        conv_output_size = (input_height // 2) * (input_width // 2) * 1

        # 初始全连接层
        self.fc0 = nn.Linear(conv_output_size, out_feature_dim)

        # 动态构建全连接层
        self.fc_layers = nn.ModuleList()
        input_size = out_feature_dim + scalar_dim

        # 层归一化
        self.layer_norm = nn.LayerNorm(input_size)

        # 创建中间全连接层
        for i in range(num_fc_layers - 1):  # 减去1是因为最后一层是value_head
            self.fc_layers.append(nn.Linear(input_size, hidden_dim))
            input_size = hidden_dim  # 后续层的输入尺寸等于hidden_dim

        # 输出层
        self.value_head = nn.Linear(hidden_dim, 1)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, matrix_inputs, scalar_inputs):
        # CNN部分
        x = self.activation(self.conv1(matrix_inputs))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # 初始全连接层
        x = self.activation(self.fc0(x))
        x = torch.cat((x, scalar_inputs), dim=1)
        x = self.layer_norm(x)

        # 动态全连接层
        for layer in self.fc_layers:
            x = self.activation(layer(x))

        # 输出层
        state_value = self.value_head(x)
        return state_value

# 定义PPO代理
class PPOAgent(Config):
    def __init__(self, policy_network, value_network):
        super().__init__()  # 调用父类的构造函数
        self.policy = policy_network
        self.policy_old = copy.deepcopy(policy_network)
        self.policy_old.eval()

        self.value_network = value_network
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.parameters["ppo"]["learning_rate"])
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.parameters["ppo"]["learning_rate"])

        self.gamma = self.parameters["ppo"]["gamma"]
        self.eps_clip = self.parameters["ppo"]["clip_range"]
        self.K_epochs = self.parameters["ppo"]["n_epochs"]

        self.memory = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.policy_old.to(self.device)
        self.value_network.to(self.device)

    def select_action(self, state):
        self.policy_old.eval()
        with torch.no_grad():
            # 提取矩阵和标量输入
            matrix_inputs = torch.FloatTensor(np.array([
                state['robot_queue_list'],
                state['picker_list'],
                state['unpicked_items_list']
            ])).unsqueeze(0).to(self.device)  # (1, 3, 21, 10)

            scalar_inputs = torch.FloatTensor(np.array([state['n_robots']] + state['n_pickers_area'])).unsqueeze(0).to(self.device)  # (1, 7)

            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

            action = action.cpu().numpy()[0]       # (7,)
            log_prob = log_prob.cpu().numpy()[0]

            # 存储记忆
            self.memory.append({
                'state': state,
                'action': action,
                'logprob': log_prob,
                'reward': None,    # 环境步后填充
                'done': None,
                'next_state': None  # 环境步后填充
            })

            return action, log_prob

    def store_reward_and_next_state(self, idx, reward, done, next_state):
        """存储奖励和下一个状态"""
        self.memory[idx]['reward'] = reward
        self.memory[idx]['done'] = done
        self.memory[idx]['next_state'] = next_state

    def compute_gae(self, rewards, values, next_values, dones):
        """
        计算广义优势估计 (Generalized Advantage Estimation)
        Args:
            rewards: 奖励序列 [T]
            values: 状态值估计 [T]
            next_values: 下一个状态值估计 [T]
            dones: 终止标志 [T]
            gamma: 折扣因子
            lam: GAE参数
        Returns:
            advantages: 广义优势估计 [T]
            returns: 目标回报 [T]
        """
        gamma = self.gamma
        lam = 0.95
        advantages = []
        last_advantage = 0
        # 反向遍历时间步
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
        advantages = torch.tensor(advantages, device=self.device)

        # 计算目标回报
        targets = advantages + values
        return advantages, targets

    def update(self):
        """使用PPO算法更新策略和值网络"""
        if len(self.memory) == 0:
            return  # 无需更新

        # 提取记忆数据
        states = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        next_states = []

        for transition in self.memory:
            states.append(transition['state'])
            actions.append(transition['action'])
            logprobs.append(transition['logprob'])
            rewards.append(transition['reward'])
            dones.append(transition['done'])
            next_states.append(transition['next_state'])

        # 转换为张量
        matrix_inputs = torch.FloatTensor(np.array([
            [
                state['robot_queue_list'],
                state['picker_list'],
                state['unpicked_items_list']
            ] for state in states
        ])).to(self.device)  # (batch, 3, 21, 10)

        scalar_inputs = torch.FloatTensor(np.array([
            [state['n_robots']] + state['n_pickers_area']
            for state in states
        ])).to(self.device)  # (batch, 7)

        actions = torch.FloatTensor(np.array(actions)).to(self.device)  # (batch, 7)
        old_logprobs = torch.FloatTensor(logprobs).to(self.device)  # (batch,)

        # 计算值和优势
        with torch.no_grad():
            values = self.value_network(matrix_inputs, scalar_inputs).squeeze(1)  # (batch,)

            next_matrix_inputs = torch.FloatTensor(np.array([
                [
                    state['robot_queue_list'],
                    state['picker_list'],
                    state['unpicked_items_list']
                ] for state in next_states
            ])).to(self.device)
            next_scalar_inputs = torch.FloatTensor(np.array([
                [state['n_robots']] + state['n_pickers_area']
                for state in next_states
            ])).to(self.device)
            next_values = self.value_network(next_matrix_inputs, next_scalar_inputs).squeeze(1)  # (batch,)

            # 转换为张量
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)

            # 使用GAE计算优势
            advantages, targets = self.compute_gae(
                rewards=rewards_tensor,
                values=values,
                next_values=next_values,
                dones=dones_tensor  # 可配置参数
            )
            # 归一化优势
            if self.parameters["ppo"]["normalize_rewards"]:
                advantages = (advantages - advantages.min()) / (advantages.max() - advantages.min() + 1e-8)  # 防止除零错误
            # 标准化优势
            if self.parameters["ppo"]["standardize_rewards"]:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(self.K_epochs):
            mean, std = self.policy(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            logprobs_new = dist.log_prob(actions).sum(dim=1)

            ratios = torch.exp(logprobs_new - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value = self.value_network(matrix_inputs, scalar_inputs).squeeze(1)
            value_loss = nn.MSELoss()(value, targets)

            # 反向传播与优化
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # 更新旧的策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        self.memory = []


# 订单对象读取函数
def get_order_object(poisson_parameter, num_items):
    """
    :param poisson_parameter: 泊松分布参数
    :param num_items: 订单中商品数量
    :return: 读取的订单对象
    """
    file_order = 'D:/Python project/DRL_Warehouse/data/instances'
    with open(file_order + "/orders_{}_{}.pkl".format(poisson_parameter, num_items), "rb") as f:
        orders = pickle.load(f)  # 读取订单数据
    return orders


# 定义训练函数
def train_ppo_agent(ppo_agent, warehouse, orders_test, num_episodes=1000):
    total_cost = float('inf')  # 初始化总成本
    train_env = copy.deepcopy(warehouse)  # 训练环境
    test_env = copy.deepcopy(warehouse)  # 测试环境
    for episode in range(352, num_episodes):
        # ====================训练=============================
        # total_seconds = 31 * 8 * 3600  # 31天
        # generate_orders = GenerateData(warehouse, total_seconds)  # 生成订单数据对象
        # orders = generate_orders.generate_orders()  # 生成一个月内的订单数据
        poisson_parameter = random.choice([60, 120, 180])  # 随机选择泊松分布参数
        num_items = random.choice([10, 20, 30])  # 随机选择订单中商品数量
        orders = get_order_object(poisson_parameter, num_items)  # 读取订单对象
        state = train_env.reset(orders)  # 重置环境并获取初始状态
        done = False
        first_step = True
        while not done:
            action, log_prob = ppo_agent.select_action(state)
            next_state, reward, done = train_env.step(action, first_step)
            ppo_agent.store_reward_and_next_state(len(ppo_agent.memory) - 1, reward, done, next_state)
            state = next_state
            first_step = False
        # 更新网络
        ppo_agent.update()

        # ===================测试==============================
        with torch.no_grad():
            state = test_env.reset(orders_test) # 重置环境并获取初始状态
            done = False
            first_step = True
            total_reward = 0
            while not done:
                action, log_prob = ppo_agent.select_action(state)
                next_state, reward, done = test_env.step(action, first_step)
                ppo_agent.store_reward_and_next_state(len(ppo_agent.memory) - 1, reward, done, next_state)
                state = next_state
                total_reward += reward
                first_step = False

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        # 可视化训练数据
        viz.line([-total_reward], [episode + 1], win='reward_III', update='append')

        # 保存模型
        if total_cost >= -total_reward:
            torch.save(ppo_agent.policy.state_dict(), f"policy_network_PPO_III.pth")
            torch.save(ppo_agent.value_network.state_dict(), f"value_network_PPO_III.pth")
            total_cost = - total_reward

        # 保存训练数据
        with open('training_data_PPO_III.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward])


if __name__ == "__main__":
    # 订单数据保存和读取位置
    file_order = 'D:/Python project/DRL_Warehouse/data/instances'
    poisson_parameter = 120  # 测试算例泊松分布参数
    num_items = 20  # 订单中的商品数量
    # 读取一个月内的订单数据，orders.pkl文件中
    with open(file_order + "/orders_{}_{}.pkl".format(poisson_parameter, num_items), "rb") as f:
        orders_test = pickle.load(f)  # 读取订单数据

    # 一个月的总秒数
    total_seconds = (8 * 3600) * 30  # 30天
    # 基于上述一个月内的订单数据和仓库环境数据，实现仓库环境的仿真
    warehouse.total_time = total_seconds  # 仿真总时间

    # 初始化网络
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()

    # 加载预训练模型参数
    policy_network.load_state_dict(torch.load("policy_network_PPO_III.pth"))
    value_network.load_state_dict(torch.load("value_network_PPO_III.pth"))

    # 初始化PPO代理
    ppo_agent = PPOAgent(policy_network, value_network)
    # 训练PPO代理
    train_ppo_agent(ppo_agent, warehouse, orders_test, num_episodes=1000)
