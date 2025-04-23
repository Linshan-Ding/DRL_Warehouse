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

# -----------------初始化仓库环境---------------------
N_l = 10  # 单个货架中储货位的数量
area_dict = {'area1': 3, 'area2': 3}  # 仓库中每个区域包含的巷道数量
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

# # 订单到达泊松分布参数
# poisson_parameter = 60  # 泊松分布参数, 60秒一个订单到达
# # 生成一个月内的订单数据，并保存到orders.pkl文件中
# generate_orders = GenerateData(warehouse, total_seconds, poisson_parameter)  # 生成订单数据对象
# generate_orders.generate_orders()  # 生成一个月内的订单数据

# 读取一个月内的订单数据，orders.pkl文件中
with open("orders.pkl", "rb") as f:
    orders = pickle.load(f)  # 读取订单数据

# 基于上述一个月内的订单数据和仓库环境数据，实现仓库环境的仿真
warehouse.total_time = total_seconds # 仿真总时间

# -----------------定义神经网络---------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=6, input_width=10, scalar_dim=3, hidden_dim=128, output_dim=3):
        """
        Policy Network with CNN for matrix inputs and FC for scalar inputs.

        Args:
            input_channels (int): Number of channels for CNN (3 matrices).
            input_height (int): Height of each matrix.
            input_width (int): Width of each matrix.
            scalar_dim (int): Number of scalar state features.
            hidden_dim (int): Number of hidden units in FC layers.
            output_dim (int): Number of actions (robot adjust, picker1 adjust, picker2 adjust).
        """
        super(PolicyNetwork, self).__init__()

        # Define CNN layers for the three matrices combined as multi-channel input
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after convolution and pooling
        conv_output_size = (input_height // 2) * (input_width // 2) * 32  # After two conv + pool layers

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + scalar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(output_dim))  # Learnable log standard deviation

    def forward(self, matrix_inputs, scalar_inputs):
        """
        Forward pass through the network.

        Args:
            matrix_inputs (torch.Tensor): Tensor of shape (batch_size, 3, 6, 10).
            scalar_inputs (torch.Tensor): Tensor of shape (batch_size, 3).

        Returns:
            mean (torch.Tensor): Mean of action distributions.
            std (torch.Tensor): Standard deviation of action distributions.
        """
        x = torch.relu(self.conv1(matrix_inputs))  # (batch_size, 16, 6, 10)
        x = torch.relu(self.conv2(x))  # (batch_size, 32, 6, 10)
        x = self.pool(x)  # (batch_size, 32, 3, 5)
        x = x.view(x.size(0), -1)  # (batch_size, 32*3*5) = (batch_size, 480)

        # Concatenate with scalar inputs
        x = torch.cat((x, scalar_inputs), dim=1)  # (batch_size, 480 + 3) = (batch_size, 483)
        x = torch.relu(self.fc1(x))  # (batch_size, hidden_dim)
        x = torch.relu(self.fc2(x))  # (batch_size, hidden_dim)
        mean = self.action_mean(x)  # (batch_size, output_dim)
        std = torch.exp(self.action_log_std)  # (output_dim,)
        std = std.expand_as(mean)  # (batch_size, output_dim)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, input_channels=3, input_height=6, input_width=10, scalar_dim=3, hidden_dim=128):
        """
        Value Network with CNN for matrix inputs and FC for scalar inputs.

        Args:
            input_channels (int): Number of channels for CNN (3 matrices).
            input_height (int): Height of each matrix.
            input_width (int): Width of each matrix.
            scalar_dim (int): Number of scalar state features.
            hidden_dim (int): Number of hidden units in FC layers.
        """
        super(ValueNetwork, self).__init__()

        # Define CNN layers for the three matrices combined as multi-channel input
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after convolution and pooling
        conv_output_size = (input_height // 2) * (input_width // 2) * 32  # After two conv + pool layers

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + scalar_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, matrix_inputs, scalar_inputs):
        """
        Forward pass through the network.

        Args:
            matrix_inputs (torch.Tensor): Tensor of shape (batch_size, 3, 6, 10).
            scalar_inputs (torch.Tensor): Tensor of shape (batch_size, 3).

        Returns:
            state_value (torch.Tensor): Estimated value of the state.
        """
        x = torch.relu(self.conv1(matrix_inputs))  # (batch_size, 16, 6, 10)
        x = torch.relu(self.conv2(x))  # (batch_size, 32, 6, 10)
        x = self.pool(x)  # (batch_size, 32, 3, 5)
        x = x.view(x.size(0), -1)  # (batch_size, 32*3*5) = (batch_size, 480)

        # Concatenate with scalar inputs
        x = torch.cat((x, scalar_inputs), dim=1)  # (batch_size, 480 + 3) = (batch_size, 483)
        x = torch.relu(self.fc1(x))  # (batch_size, hidden_dim)
        x = torch.relu(self.fc2(x))  # (batch_size, hidden_dim)
        state_value = self.value_head(x)  # (batch_size, 1)
        return state_value

# -----------------PPO算法实现-------------------------
class PPOAgent:
    def __init__(self, policy_network, value_network, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        """
        PPO Agent with separate policy and value networks.

        Args:
            policy_network (nn.Module): The policy network.
            value_network (nn.Module): The value network.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            eps_clip (float): Clipping parameter.
            K_epochs (int): Number of training epochs.
        """
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
        """
        Select an action based on the current state.

        Args:
            state (dict): Current state dictionary.

        Returns:
            action_tuple (tuple): (robot_adjust, picker1_adjust, picker2_adjust)
            log_prob (float): Log probability of the action.
        """
        self.policy_old.eval()
        with torch.no_grad():
            # Extract matrix and scalar inputs
            matrix_inputs = torch.FloatTensor(np.array([
                np.array(state['robot_queue_list']),
                np.array(state['picker_list']),
                np.array(state['unpicked_items_list'])])).unsqueeze(0).to(self.device)  # Shape: (1, 3, 6, 10)
            scalar_inputs = torch.FloatTensor(np.array([
                state['n_robots_at_depot'],
                state['n_robots'],
                state['n_pickers']])).unsqueeze(0).to(self.device)  # Shape: (1, 3)

            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=1)

            action = action.cpu().numpy()[0]
            action_logprob = action_logprob.cpu().numpy()[0]

            # Clip actions to a reasonable range, e.g., [-5, 5]
            action = np.clip(action, -5, 5)

            # 动作中的每个元素取整
            action = np.round(action).astype(int)

            # Store information for PPO update
            self.memory.append({
                'state': state,
                'action': action,
                'logprob': action_logprob,
                'reward': None,  # To be filled after environment step
                'done': None,
                'next_state': None  # To be filled after environment step
            })

            # Convert to floats for continuous action space
            return action, action_logprob

    def store_reward_and_next_state(self, idx, reward, done, next_state):
        """
        Store reward, done flag, and next state in memory at index idx.

        Args:
            idx (int): Index in memory.
            reward (float): Reward obtained.
            done (bool): Whether the episode is done.
            next_state (dict): The next state after taking the action.
        """
        self.memory[idx]['reward'] = reward
        self.memory[idx]['done'] = done
        self.memory[idx]['next_state'] = next_state

    def update(self):
        """
        Update policy and value networks using PPO algorithm.
        """
        # Convert memory to tensors
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

        # Prepare tensors with numpy intermediate conversion
        matrix_inputs = torch.FloatTensor(np.array([
            [
                np.array(state['robot_queue_list']),
                np.array(state['picker_list']),
                np.array(state['unpicked_items_list'])
                    ] for state in states
        ])).to(self.device)  # Shape: (batch, 3, 6, 10)

        scalar_inputs = torch.FloatTensor(np.array([
                    [
                        state['n_robots_at_depot'],
                        state['n_robots'],
                        state['n_pickers']
                    ] for state in states
        ])).to(self.device)  # Shape: (batch, 3)

        actions = torch.FloatTensor(np.array(actions)).to(self.device)  # Shape: (batch, 3)
        old_logprobs = torch.FloatTensor(logprobs).to(self.device)  # Shape: (batch,)

        # Compute values and advantages
        with torch.no_grad():
            # Value estimates for current states
            values = self.value_network(matrix_inputs, scalar_inputs).squeeze(1)  # Shape: (batch,)
            # Value estimates for next states
            next_matrix_inputs = torch.FloatTensor(np.array([
                [
                    np.array(state['robot_queue_list']),
                    np.array(state['picker_list']),
                    np.array(state['unpicked_items_list'])
                            ] for state in next_states
            ])).to(self.device)  # Shape: (batch, 3, 6, 10)

            next_scalar_inputs = torch.FloatTensor(np.array([
                            [
                                state['n_robots_at_depot'],
                                state['n_robots'],
                                state['n_pickers']
                            ] for state in next_states
            ])).to(self.device)  # Shape: (batch, 3)

            next_values = self.value_network(next_matrix_inputs, next_scalar_inputs).squeeze(1)  # Shape: (batch,)
            # Compute targets
            targets = torch.FloatTensor(rewards).to(self.device) + self.gamma * next_values * (1 - torch.FloatTensor(dones).to(self.device))
            advantages = targets - values

        # Optimize policy and value networks
        for _ in range(self.K_epochs):
            # Recompute means and stds
            mean, std = self.policy(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            logprobs_new = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)

            # Compute ratio
            ratios = torch.exp(logprobs_new - old_logprobs)

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(self.value_network(matrix_inputs, scalar_inputs).squeeze(1), targets)
            entropy_loss = -0.01 * entropy.mean()

            loss = policy_loss + 0.5 * value_loss + entropy_loss

            # Take gradient step
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.memory = []

# -----------------训练PPO---------------------
def train_ppo_agent(ppo_agent, warehouse_env, num_episodes=1000):
    """
    Train the PPO agent in the warehouse environment.

    Args:
        ppo_agent (PPOAgent): The PPO agent.
        warehouse_env (WarehouseEnv): The warehouse environment.
        num_episodes (int): Number of training episodes.
    """
    for episode in range(num_episodes):
        state = warehouse_env.reset(orders)
        done = False
        total_reward = 0

        while not done:
            action, log_prob = ppo_agent.select_action(state)
            next_state, reward, done = warehouse_env.step(action)
            # Store reward, done flag, and next state
            ppo_agent.store_reward_and_next_state(len(ppo_agent.memory) - 1, reward, done, next_state)
            state = next_state
            total_reward += reward

        # Update PPO agent after each episode
        ppo_agent.update()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# -----------------主函数---------------------
if __name__ == "__main__":
    # Initialize networks
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()

    # Initialize PPO agent
    ppo_agent = PPOAgent(policy_network, value_network)

    # Train the PPO agent
    train_ppo_agent(ppo_agent, warehouse, num_episodes=1000)