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

import multiprocessing as mp
import cloudpickle
import copy


def worker(remote, parent_remote, env_fn_wrapper, warehouse):
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            state, reward, done = env.step(data)
            if done:
                # 自动生成新订单并 reset
                total_seconds = 31 * 8 * 3600
                generate_orders = GenerateData(warehouse, total_seconds)
                orders = generate_orders.generate_orders()
                state = env.reset(orders)
            remote.send((state, reward, done))

        elif cmd == "reset":
            # 每个子进程独立生成订单
            total_seconds = 31 * 8 * 3600
            generate_orders = GenerateData(warehouse, total_seconds)
            orders = generate_orders.generate_orders()
            state = env.reset(orders)
            remote.send(state)

        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    用于序列化环境构造函数，避免 lambda/env_fn 在多进程中无法传递的问题
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)


class SubprocVecEnv:
    def __init__(self, env_fns, warehouse):
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.ps = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn), warehouse)
            )
            p.daemon = True
            p.start()
            self.ps.append(p)
            work_remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        states, rewards, dones = zip(*results)
        return list(states), list(rewards), list(dones)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()

# 设置训练数据可视化
viz = Visdom(env='PPO_I')
viz.line([0], [0], win='reward_I', opts=dict(title='Reward1', xlabel='Episode', ylabel='Reward'))

# 初始化仓库环境
warehouse = WarehouseEnv()
N_w = warehouse.N_w # 仓库巷道数量
N_l = warehouse.N_l # 单个货架中储货位数量
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
                state['robot_queue_list'],
                state['picker_list'],
                state['unpicked_items_list']
            ])).unsqueeze(0).to(self.device)  # (1, 3, 21, 10)

            scalar_inputs = torch.FloatTensor(np.array([state['n_robots']] + state['n_pickers_area'])).unsqueeze(0).to(self.device)  # (1, 7)

            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

            action = action.cpu().numpy()[0]      # (7,)
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
            # 获取所有状态的值估计
            values = self.value_network(matrix_inputs, scalar_inputs).squeeze(1)

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
            ])).to(self.device)
            next_values = self.value_network(next_matrix_inputs, next_scalar_inputs).squeeze(1)

            # 转换为张量
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)

            # 使用GAE计算优势
            advantages, targets = self.compute_gae(
                rewards=rewards_tensor,
                values=values,
                next_values=next_values,
                dones=dones_tensor # 可配置参数
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
            # 计算熵
            entropy = dist.entropy().mean()

            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value = self.value_network(matrix_inputs, scalar_inputs).squeeze(1)
            value_loss = nn.MSELoss()(value, targets)
            # 计算熵损失
            entropy_loss = -self.ent_coef * entropy
            # 总损失
            # total_loss = policy_loss + 0.5 * value_loss + entropy_loss

            # 反向传播与优化
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # 在更新后衰减熵系数
        self.ent_coef = max(self.ent_coef * self.ent_coef_decay, self.min_ent_coef)

        # 更新旧的策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        self.memory = []

def train_ppo_agent(ppo_agent, warehouse, orders_test, num_episodes=1000, n_envs=8):
    # 创建并行训练环境
    envs = SubprocVecEnv(
        [lambda: copy.deepcopy(warehouse) for _ in range(n_envs)],
        warehouse=warehouse
    )
    total_cost = float('inf')

    for episode in range(num_episodes):
        # =================== 并行训练 ===================
        states = envs.reset()
        dones = [False] * n_envs
        total_rewards = [0] * n_envs

        while not all(dones):
            actions, log_probs = [], []
            for state in states:
                action, log_prob = ppo_agent.select_action(state)
                actions.append(action)
                log_probs.append(log_prob)

            next_states, rewards, dones = envs.step(actions)

            for i in range(n_envs):
                ppo_agent.store_reward_and_next_state(
                    len(ppo_agent.memory) - n_envs + i,
                    rewards[i], dones[i], next_states[i]
                )
                total_rewards[i] += rewards[i]

            states = next_states

        avg_reward = sum(total_rewards) / n_envs
        ppo_agent.update()
        print(f"[Train] Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

        # =================== 测试评估 ===================
        with torch.no_grad():
            test_env = copy.deepcopy(warehouse)
            state = test_env.reset(orders_test)  # 使用固定订单文件
            done = False
            test_reward = 0
            while not done:
                action, _ = ppo_agent.select_action(state)
                next_state, reward, done = test_env.step(action)
                state = next_state
                test_reward += reward

        print(f"[Test ] Episode {episode + 1}/{num_episodes}, Total Test Reward: {test_reward:.2f}")
        viz.line([-test_reward], [episode + 1], win='reward_I', update='append')

        # =================== 保存模型 ===================
        if total_cost >= -test_reward:  # 用测试奖励作为评估指标
            torch.save(ppo_agent.policy.state_dict(), "policy_network_PPO_I.pth")
            torch.save(ppo_agent.value_network.state_dict(), "value_network_PPO_I.pth")
            total_cost = -test_reward

    envs.close()



if __name__ == "__main__":
    file_order = 'D:\\Python project\\DRL_Warehouse\\data'
    poisson_parameter = 120
    with open(file_order + "\\orders_{}.pkl".format(poisson_parameter), "rb") as f:
        orders_test = pickle.load(f)

    total_seconds = 30 * 8 * 3600
    warehouse.total_time = total_seconds

    policy_network = PolicyNetwork()
    value_network = ValueNetwork()
    ppo_agent = PPOAgent(policy_network, value_network)

    # 训练+周期性测试
    train_ppo_agent(ppo_agent, warehouse, orders_test, num_episodes=1000, n_envs=3)
