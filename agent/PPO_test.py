"""
PPO agent：Proximal Policy Optimization
调整每个决策点机器人和拣货员的数量：拣货员+日租
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from environment.warehouse_I import WarehouseEnv  # 导入仓库环境类
import numpy as np
import copy
import pickle # 用于读取订单数据
from data.generat_order import GenerateData
from visdom import Visdom
import csv
from environment.class_public import Config
import random

# 设置训练数据可视化
viz = Visdom(env='PPO_I')
viz.line([0], [0], win='reward_I', opts=dict(title='Reward1', xlabel='Episode', ylabel='Reward'))

# 初始化仓库环境
warehouse = WarehouseEnv()
N_w = warehouse.N_w # 仓库巷道数量
N_l = warehouse.N_l # 单个货架中储货位数量
N_a = warehouse.N_a # 仓库区域数量


class ActionAttention(nn.Module):
    def __init__(self, embed_dim=128, heads=4):
        super().__init__()
        self.embed = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """输入x: [batch, action_dim]"""
        embedded = self.embed(x)  # [batch, embed_dim]
        attn_out, _ = self.attn(
            embedded.unsqueeze(1),  # [batch, 1, embed]
            embedded.unsqueeze(1),
            embedded.unsqueeze(1)
        )
        return self.norm(attn_out.squeeze(1))

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=1, input_height=N_w, input_width=N_l,
                 scalar_dim=N_a + 1, hidden_dim=128, output_dim=N_a + 1,
                 out_feature_dim=1, fc_layers=10):
        super(PolicyNetwork, self).__init__()
        self.output_dim = output_dim
        self.fc_layers = fc_layers  # 存储全连接层数量
        self.action_attn = ActionAttention()

        # CNN层
        self.conv1 = nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入尺寸
        conv_output_size = (input_height // 2) * (input_width // 2) * 1  # 9 * 5 * 1 = 45

        # 动态构建全连接层
        self.fc0 = nn.Linear(conv_output_size, out_feature_dim)  # (batch_size, 1)

        # 创建全连接层列表
        self.fc_modules = nn.ModuleList()
        input_size = out_feature_dim + scalar_dim

        # 层归一化
        self.layer_norm = nn.LayerNorm(input_size)

        # 根据fc_layers参数动态构建隐藏层
        for i in range(fc_layers):
            self.fc_modules.append(nn.Linear(input_size if i == 0 else hidden_dim, hidden_dim))
        # 输出层
        self.action_mean = nn.Linear(hidden_dim*2, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(output_dim))

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, matrix_inputs, scalar_inputs):
        x = self.activation(self.conv1(matrix_inputs))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # 拼接标量输入
        x = self.activation(self.fc0(x))
        x = torch.cat((x, scalar_inputs), dim=1)
        x = self.layer_norm(x)

        # 动态处理全连接层
        for fc_layer in self.fc_modules:
            x = self.activation(fc_layer(x))
        attn_x = self.action_attn(x)
        fused = torch.cat([x, attn_x], dim=1)
        mean = self.action_mean(fused)
        std = torch.exp(self.action_log_std)
        std = std.expand_as(mean)
        return mean, std

# 定义值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_channels=1, input_height=N_w, input_width=N_l,
                 scalar_dim=N_a + 1, hidden_dim=128, out_feature_dim=1,
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
        # 添加动态熵系数控制
        self.ent_coef = self.parameters['ppo']['initial_entropy_coeff']  # 初始熵系数
        self.ent_coef_decay = self.parameters['ppo']['entropy_coeff_decay']  # 熵衰减率
        self.min_ent_coef = self.parameters['ppo']['min_entropy_coeff']  # 最小熵系数
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
                state['unpicked_items_list']
            ])).unsqueeze(0).to(self.device)  # (1, 3, 21, 10)

            scalar_inputs = torch.FloatTensor(np.array([state['n_robots']] + state['n_pickers_area'])).unsqueeze(0).to(self.device)  # (1, 7)

            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

            action = action.cpu().numpy()[0]      # (7,)
            log_prob = log_prob.cpu().numpy()[0]

            # 动作范围限制
            print('action:', action)

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
            # 基于reward计算累计折扣回报作为targets
            discounted_returns = [0]   # 初始化折扣回报列表
            # 计算折扣回报
            for ix in range(len(states)):
                return_value = rewards[-(ix + 1)] + self.gamma * discounted_returns[-1]
                discounted_returns.append(return_value)
            discounted_returns = discounted_returns[1:]  # 去掉第一个元素
            discounted_returns = discounted_returns[::-1]  # 反转列表
            discounted_returns = torch.tensor(discounted_returns).to(self.device).detach()  # (batch,)
            # 回报归一化
            if self.parameters["ppo"]["normalize_rewards"]:
                discounted_returns = (discounted_returns - discounted_returns.min()) / (discounted_returns.max() - discounted_returns.min() + 1e-8)  # 防止除零错误
            # 回报标准化
            if self.parameters["ppo"]["standardize_rewards"]:
                discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)  # 防止除零错误
            # discounted_returns数据类型转为torch.float32
            discounted_returns = discounted_returns.float().to(self.device)  # (batch,)
            # 计算优势
            advantages = discounted_returns - values  # (batch,)

        # PPO更新
        for _ in range(self.K_epochs):
            mean, std = self.policy(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            logprobs_new = dist.log_prob(actions).sum(dim=1)
            ratios = torch.exp(logprobs_new - old_logprobs)
            # 计算熵
            entropy = dist.entropy().mean()

            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value = self.value_network(matrix_inputs, scalar_inputs).squeeze(1)
            value_loss = nn.MSELoss()(value, discounted_returns)
            # 计算熵损失
            entropy_loss = -self.ent_coef * entropy
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss + entropy_loss

            # 反向传播与优化
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # 在更新后衰减熵系数
        self.ent_coef = max(self.ent_coef * self.ent_coef_decay, self.min_ent_coef)

        # 更新旧的策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        self.memory = []

# 定义训练函数
def train_ppo_agent(ppo_agent, warehouse_env, num_episodes=1000):
    for episode in range(num_episodes):
        state = warehouse_env.reset(orders) # 重置环境并获取初始状态
        done = False
        total_reward = 0

        while not done:
            action, log_prob = ppo_agent.select_action(state)
            next_state, reward, done = warehouse_env.step(action)
            ppo_agent.store_reward_and_next_state(len(ppo_agent.memory) - 1, reward, done, next_state)
            state = next_state
            total_reward += reward
            # print(done)
            print(state['n_robots'], state['n_pickers_area'])
            print('完成的订单数量：', len(warehouse.orders_completed))

        ppo_agent.update()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        # 可视化训练数据
        viz.line([-total_reward], [episode + 1], win='reward_I', update='append')

        # 保存模型
        if (episode + 1) % 500 == 0:
            torch.save(ppo_agent.policy.state_dict(), f"policy_network_PPO_I_{episode + 1}.pth")
            torch.save(ppo_agent.value_network.state_dict(), f"value_network_PPO_I_{episode + 1}.pth")

        # 保存训练数据
        with open('training_data_PPO_I.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward])


if __name__ == "__main__":
    # 订单数据保存和读取位置
    file_order = 'D:\Python project\DRL_Warehouse\data'
    poisson_parameter = 60  # 泊松分布参数, 60秒一个订单到达
    # 一个月的总秒数
    total_seconds = 3 * 8 * 3600  # 6天
    # 基于上述一个月内的订单数据和仓库环境数据，实现仓库环境的仿真
    warehouse.total_time = total_seconds  # 仿真总时间

    # # 生成一个月内的订单数据，并保存到orders.pkl文件中
    # generate_orders = GenerateData(warehouse, total_seconds, poisson_parameter)  # 生成订单数据对象
    # orders = generate_orders.generate_orders()  # 生成一个月内的订单数据

    # 读取一个月内的订单数据，orders.pkl文件中
    with open(file_order + "\orders_{}.pkl".format(poisson_parameter), "rb") as f:
        orders = pickle.load(f)  # 读取订单数据

    # 初始化网络
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()
    # 初始化PPO代理
    ppo_agent = PPOAgent(policy_network, value_network)
    # 训练PPO代理
    train_ppo_agent(ppo_agent, warehouse, num_episodes=1000)