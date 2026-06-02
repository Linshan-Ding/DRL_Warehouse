from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.training_utils import (  # noqa: E402
    DEFAULT_PARAMETERS,
    CsvLogger,
    best_config_path as _best_config_path,
    case_stem,
    collect_resource_config,
    collect_metrics,
    init_base_env as _init_base_env,
    layer_init,
    load_orders as _load_orders,
    make_episode_env,
    make_visdom as _make_visdom,
    max_decisions,
    output_paths as _output_paths,
    print_episode,
    save_checkpoint,
    set_seed,
    step_env,
    write_best_config_csv,
)


MODES = ("short", "long", "hybrid")
ITEM_SCENARIOS = tuple(DEFAULT_PARAMETERS["experiment"].get("item_scenarios", [2, 4, 6, 10]))
MONTHS = tuple(DEFAULT_PARAMETERS["experiment"].get("months", list(range(1, 13))))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    experiment = DEFAULT_PARAMETERS["experiment"]
    paths = DEFAULT_PARAMETERS["paths"]
    parser.add_argument("--mode", choices=MODES, default=experiment.get("mode", "short"))
    parser.add_argument("--items", type=int, choices=ITEM_SCENARIOS, default=experiment.get("item_scenario", 2))
    parser.add_argument("--month", type=int, choices=MONTHS, default=experiment.get("month", 1))
    parser.add_argument("--episodes", type=int, default=experiment.get("episodes", 3000))
    parser.add_argument("--seed", type=int, default=experiment.get("seed", 0))
    parser.add_argument("--output-dir", default=paths.get("baseline_output_dir", "result/baselines"))
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("--action-scale", type=int, default=experiment.get("action_scale", 5))
    parser.add_argument("--max-days", type=int, default=experiment.get("max_days", 30))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def output_paths(args: argparse.Namespace, algorithm: str) -> tuple[Path, Path]:
    stem = f"{algorithm}_{case_stem(args.mode, args.items, args.month, args.seed)}"
    return _output_paths(args.output_dir, stem)


def best_config_path(args: argparse.Namespace, algorithm: str) -> Path:
    stem = f"{algorithm}_{case_stem(args.mode, args.items, args.month, args.seed)}"
    return _best_config_path(args.output_dir, stem)


load_orders = partial(_load_orders, DEFAULT_PARAMETERS)
init_base_env = partial(_init_base_env, DEFAULT_PARAMETERS)


def make_visdom(args: argparse.Namespace, algorithm: str):
    win = f"{algorithm}_{case_stem(args.mode, args.items, args.month, args.seed)}"
    return _make_visdom(args.visdom, "DRL_Baselines", win)


def state_to_arrays(state: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack(
        [
            state["robot_queue_list"],
            state["picker_list"],
            state["unpicked_items_list"],
        ],
        axis=0,
    ).astype(np.float32)
    scalar = np.asarray([state["n_robots"]] + list(state["n_pickers_area"]), dtype=np.float32)
    return matrix, scalar


def state_to_tensors(state: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix, scalar = state_to_arrays(state)
    return (
        torch.as_tensor(matrix, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(scalar, dtype=torch.float32, device=device).unsqueeze(0),
    )


def normalized_to_env_action(action: np.ndarray, action_scale: int) -> np.ndarray:
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    scaled = np.rint(clipped * action_scale)
    return np.clip(scaled, -action_scale, action_scale).astype(np.float32)


class StateEncoder(nn.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        scalar_dim: int,
        hidden_dim: int = 128,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_height, input_width)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.visual_fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, feature_dim)),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(feature_dim + scalar_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

    def forward(self, matrix: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        visual = self.visual_fc(self.cnn(matrix))
        return self.backbone(torch.cat([visual, scalar], dim=-1))


class A2CActor(nn.Module):
    def __init__(self, input_height: int, input_width: int, scalar_dim: int, action_dim: int):
        super().__init__()
        self.encoder = StateEncoder(input_height, input_width, scalar_dim)
        self.mean = layer_init(nn.Linear(128, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, matrix: torch.Tensor, scalar: torch.Tensor):
        hidden = self.encoder(matrix, scalar)
        mean = self.mean(hidden)
        std = self.log_std.expand_as(mean).exp()
        return mean, std


class DeterministicActor(nn.Module):
    def __init__(self, input_height: int, input_width: int, scalar_dim: int, action_dim: int):
        super().__init__()
        self.encoder = StateEncoder(input_height, input_width, scalar_dim)
        self.head = layer_init(nn.Linear(128, action_dim), std=0.01)

    def forward(self, matrix: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.encoder(matrix, scalar)))


class SquashedGaussianActor(nn.Module):
    def __init__(self, input_height: int, input_width: int, scalar_dim: int, action_dim: int):
        super().__init__()
        self.encoder = StateEncoder(input_height, input_width, scalar_dim)
        self.mean = layer_init(nn.Linear(128, action_dim), std=0.01)
        self.log_std = layer_init(nn.Linear(128, action_dim), std=0.01)

    def forward(self, matrix: torch.Tensor, scalar: torch.Tensor):
        hidden = self.encoder(matrix, scalar)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -20, 2)
        return mean, log_std

    def sample(self, matrix: torch.Tensor, scalar: torch.Tensor):
        mean, log_std = self.forward(matrix, scalar)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, matrix: torch.Tensor, scalar: torch.Tensor):
        mean, _ = self.forward(matrix, scalar)
        return torch.tanh(mean)


class ValueNetwork(nn.Module):
    def __init__(self, input_height: int, input_width: int, scalar_dim: int):
        super().__init__()
        self.encoder = StateEncoder(input_height, input_width, scalar_dim)
        self.value = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, matrix: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self.value(self.encoder(matrix, scalar))


class QNetwork(nn.Module):
    def __init__(self, input_height: int, input_width: int, scalar_dim: int, action_dim: int):
        super().__init__()
        self.encoder = StateEncoder(input_height, input_width, scalar_dim)
        self.q = nn.Sequential(
            layer_init(nn.Linear(128 + action_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward(self, matrix: torch.Tensor, scalar: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(matrix, scalar)
        return self.q(torch.cat([hidden, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int, matrix_shape: Tuple[int, ...], scalar_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.matrix = np.zeros((capacity, *matrix_shape), dtype=np.float32)
        self.scalar = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_matrix = np.zeros((capacity, *matrix_shape), dtype=np.float32)
        self.next_scalar = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        matrix: np.ndarray,
        scalar: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_matrix: np.ndarray,
        next_scalar: np.ndarray,
        done: bool,
    ) -> None:
        self.matrix[self.ptr] = matrix
        self.scalar[self.ptr] = scalar
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_matrix[self.ptr] = next_matrix
        self.next_scalar[self.ptr] = next_scalar
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.matrix[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.scalar[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.action[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_matrix[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_scalar[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
        )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - tau).add_(tau * src_param.data)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
