"""
PPO agent：Proximal Policy Optimization
调整每个决策点机器人和拣货员的数量
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from environment.warehouse import WarehouseEnv  # 导入仓库环境类
import numpy as np
import copy
import pickle # 用于读取订单数据
from data.generat_order import GenerateData

# -----------------初始化仓库环境---------------------
N_l = 10  # 单个货架中储货位的数量
area_dict = {'area1': 3, 'area2': 3, 'area3': 3, 'area4': 3, 'area5': 3, 'area6': 3}  # 仓库中每个区域包含的巷道数量
N_w = sum(area_dict.values())  # 巷道的数量
area_ids = list(area_dict.keys())
S_l = 1  # 储货位的长度
S_w = 1  # 储货位的宽度
S_b = 2  # 底部通道的宽度
S_d = 2  # 仓库的出入口处的宽度
S_a = 2  # 巷道的宽度
depot_position = (0, 0)  # 机器人的起始位置
# 初始化仓库环境
warehouse = WarehouseEnv(N_l, N_w, S_l, S_w, S_b, S_d, S_a, area_dict, area_ids, depot_position)
# 一个月的总秒数
total_seconds = 6 * 24 * 3600  # 7天

# 订单数据保存和读取位置
file_order = 'D:\Python project\DRL_Warehouse\data'
# 订单到达泊松分布参数
poisson_parameter = 60  # 泊松分布参数, 60秒一个订单到达

# # 生成一个月内的订单数据，并保存到orders.pkl文件中
# generate_orders = GenerateData(warehouse, total_seconds, poisson_parameter)  # 生成订单数据对象
# generate_orders.generate_orders()  # 生成一个月内的订单数据

# 读取一个月内的订单数据，orders.pkl文件中
with open(file_order+ "\orders_{}.pkl".format(poisson_parameter), "rb") as f:
    orders = pickle.load(f)  # 读取订单数据

# 基于上述一个月内的订单数据和仓库环境数据，实现仓库环境的仿真
warehouse.total_time = total_seconds # 仿真总时间

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=18, input_width=10, scalar_dim=7, hidden_dim=128, output_dim=7):
        super(PolicyNetwork, self).__init__()
        self.output_dim = output_dim

        # CNN层
        self.conv1 = nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入尺寸
        conv_output_size = (input_height // 2) * (input_width // 2) * 1  # 9 * 5 * 1 = 45

        # 全连接层
        self.fc1 = nn.Linear(conv_output_size + scalar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(output_dim))

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, matrix_inputs, scalar_inputs):
        x = self.activation(self.conv1(matrix_inputs))  # (batch_size, 16, 18, 10)
        x = self.activation(self.conv2(x))             # (batch_size, 32, 18, 10)
        x = self.pool(x)                                # (batch_size, 32, 10, 5)
        x = x.view(x.size(0), -1)                      # (batch_size, 1600)

        # 拼接标量输入
        x = torch.cat((x, scalar_inputs), dim=1)       # (batch_size, 1607)
        x = self.activation(self.fc1(x))               # (batch_size, hidden_dim)
        x = self.activation(self.fc2(x))               # (batch_size, hidden_dim)
        mean = self.action_mean(x)                      # (batch_size, 7)
        std = torch.exp(self.action_log_std)            # (7,)
        std = std.expand_as(mean)                        # (batch_size, 7)
        return mean, std

# 定义值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=18, input_width=10, scalar_dim=7, hidden_dim=128):
        super(ValueNetwork, self).__init__()

        # CNN层
        self.conv1 = nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入尺寸
        conv_output_size = (input_height // 2) * (input_width // 2) * 1  # 9 * 5 * 32 = 1600

        # 全连接层
        self.fc1 = nn.Linear(conv_output_size + scalar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, matrix_inputs, scalar_inputs):
        x = self.activation(self.conv1(matrix_inputs))  # (batch_size, 16, 18, 10)
        x = self.activation(self.conv2(x))             # (batch_size, 32, 18, 10)
        x = self.pool(x)                                # (batch_size, 32, 10, 5)
        x = x.view(x.size(0), -1)                      # (batch_size, 1600)

        # 拼接标量输入
        x = torch.cat((x, scalar_inputs), dim=1)       # (batch_size, 1607)
        x = self.activation(self.fc1(x))               # (batch_size, hidden_dim)
        x = self.activation(self.fc2(x))               # (batch_size, hidden_dim)
        state_value = self.value_head(x)                # (batch_size, 1)
        return state_value

# 定义PPO代理
class PPOAgent:
    def __init__(self, policy_network, value_network, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = policy_network
        self.policy_old = copy.deepcopy(policy_network)
        self.policy_old.eval()

        self.value_network = value_network
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

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

            # 动作范围限制
            action = np.clip(action, -5, 5)

            # 如果动作需要为整数，进行四舍五入
            action = np.round(action).astype(int)

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
            ])).to(self.device)  # (batch, 3, 21, 10)

            next_scalar_inputs = torch.FloatTensor(np.array([
                [state['n_robots']] + state['n_pickers_area']
                for state in next_states
            ])).to(self.device)  # (batch, 7)

            next_values = self.value_network(next_matrix_inputs, next_scalar_inputs).squeeze(1)  # (batch,)

            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)
            targets = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
            advantages = targets - values

        # PPO更新
        for _ in range(self.K_epochs):
            mean, std = self.policy(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            logprobs_new = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)

            ratios = torch.exp(logprobs_new - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value = self.value_network(matrix_inputs, scalar_inputs).squeeze(1)
            value_loss = nn.MSELoss()(value, targets)

            entropy_loss = -0.01 * entropy.mean()

            loss = policy_loss + 0.5 * value_loss + entropy_loss

            # 反向传播与优化
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # 更新旧的策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        self.memory = []

# 定义训练函数
def train_ppo_agent(ppo_agent, warehouse_env, num_episodes=1000):
    for episode in range(num_episodes):
        state = warehouse_env.reset(orders)
        done = False
        total_reward = 0

        while not done:
            action, log_prob = ppo_agent.select_action(state)
            next_state, reward, done = warehouse_env.step(action)
            ppo_agent.store_reward_and_next_state(len(ppo_agent.memory) - 1, reward, done, next_state)
            state = next_state
            total_reward += reward

        ppo_agent.update()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


if __name__ == "__main__":
    # 初始化网络
    policy_network = PolicyNetwork(output_dim=7)
    value_network = ValueNetwork()
    # 初始化PPO代理
    ppo_agent = PPOAgent(policy_network, value_network)
    # 训练PPO代理
    train_ppo_agent(ppo_agent, warehouse, num_episodes=1000)
