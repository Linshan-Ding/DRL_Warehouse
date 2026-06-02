from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import BatchSampler, SubsetRandomSampler

from agent.ppo.buffer import RolloutBuffer
from environment.class_public import Config


class PPOAgent(Config):
    def __init__(self, policy_network, value_network, parameters=None, device: str | None = None):
        super().__init__()
        if parameters is not None:
            self.parameters = parameters

        configured_device = device or self.parameters["experiment"].get("device", "auto")
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(configured_device)

        self.policy = policy_network.to(self.device)
        self.value_network = value_network.to(self.device)
        self.policy_old = copy.deepcopy(policy_network).to(self.device)
        self.policy_old.eval()

        ppo = self.parameters["ppo"]
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": ppo["learning_rate"]},
                {"params": self.value_network.parameters(), "lr": ppo["learning_rate"]},
            ]
        )
        self.gamma = ppo["gamma"]
        self.eps_clip = ppo["clip_range"]
        self.k_epochs = ppo["n_epochs"]
        self.gae_lambda = ppo.get("gae_lambda", 0.95)
        self.batch_size = ppo.get("batch_size", 64)
        self.ent_coef = ppo.get("initial_entropy_coeff", 0.05)
        self.ent_coef_decay = ppo.get("entropy_coeff_decay", 0.995)
        self.min_ent_coef = ppo.get("min_entropy_coeff", 0.001)
        self.max_grad_norm = ppo.get("max_grad_norm", 0.5)
        self.value_loss_coeff = ppo.get("value_loss_coeff", 0.5)
        self.mse_loss = nn.MSELoss()
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        self.policy_old.eval()
        with torch.no_grad():
            matrix_inputs = torch.as_tensor(
                np.array(
                    [
                        state["robot_queue_list"],
                        state["picker_list"],
                        state["unpicked_items_list"],
                    ]
                ),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            scalar_inputs = torch.as_tensor(
                np.array([state["n_robots"]] + state["n_pickers_area"]),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            mean, std = self.policy_old(matrix_inputs, scalar_inputs)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)
            value = self.value_network(matrix_inputs, scalar_inputs)

        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy()[0],
            value.cpu().numpy()[0],
            matrix_inputs,
            scalar_inputs,
        )

    def update(self) -> bool:
        if len(self.buffer) <= 1:
            self.buffer.clear()
            return False

        matrix_states = torch.cat(self.buffer.matrix_states).to(self.device)
        scalar_states = torch.cat(self.buffer.scalar_states).to(self.device)
        actions = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_logprobs = torch.as_tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.array(self.buffer.rewards), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array(self.buffer.dones), dtype=torch.float32, device=self.device)
        values = torch.as_tensor(np.array(self.buffer.values), dtype=torch.float32, device=self.device).squeeze()
        if values.dim() == 0:
            values = values.unsqueeze(0)

        returns = []
        advantages = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]

        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        dataset_size = len(rewards)
        batch_size = min(self.batch_size, dataset_size)
        for _ in range(self.k_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False)
            for indices in sampler:
                indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                b_matrix = matrix_states[indices]
                b_scalar = scalar_states[indices]
                b_actions = actions[indices]
                b_old_logprobs = old_logprobs[indices]
                b_returns = returns[indices].view(-1)
                b_advantages = advantages[indices]

                mean, std = self.policy(b_matrix, b_scalar)
                dist = Normal(mean, std)
                logprobs = dist.log_prob(b_actions).sum(dim=1)
                dist_entropy = dist.entropy().sum(dim=1)
                state_values = self.value_network(b_matrix, b_scalar).squeeze()
                if state_values.dim() == 0:
                    state_values = state_values.unsqueeze(0)

                ratios = torch.exp(logprobs - b_old_logprobs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                loss = (
                    -torch.min(surr1, surr2).mean()
                    + self.value_loss_coeff * self.mse_loss(state_values, b_returns)
                    - self.ent_coef * dist_entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.ent_coef = max(self.ent_coef * self.ent_coef_decay, self.min_ent_coef)
        self.buffer.clear()
        return True

