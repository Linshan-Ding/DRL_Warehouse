from __future__ import annotations

import argparse
import copy
import csv
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.warehouse_test2 import WarehouseEnv  # noqa: E402


MODES = ("short", "long", "hybrid")
SCENARIOS = (2, 4, 6, 10)
CSV_HEADER = [
    "episode",
    "total_cost",
    "delay_cost",
    "robot_cost",
    "picker_cost",
    "completed_orders",
    "on_time_completed_orders",
    "total_orders",
    "average_picking_time",
    "completion_rate",
    "scenario",
    "mode",
    "seed",
]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mode", choices=MODES, default="short")
    parser.add_argument("--scenario", type=int, choices=SCENARIOS, default=2)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="result/baselines")
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("--action-scale", type=int, default=5)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_paths(args: argparse.Namespace, algorithm: str) -> Tuple[Path, Path]:
    output_dir = resolve_output_dir(args.output_dir)
    stem = f"{algorithm}_{args.mode}_s{args.scenario}_seed{args.seed}"
    return output_dir / f"{stem}.csv", output_dir / f"{stem}.pth"


def load_orders(scenario: int):
    path = REPO_ROOT / "data" / "instances" / f"orders_{scenario}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Order file not found: {path}. Generate it first or choose an existing scenario."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def init_base_env() -> WarehouseEnv:
    env = WarehouseEnv()
    env.total_time = 8 * 3600 * 30
    return env


def state_to_arrays(state: Dict) -> Tuple[np.ndarray, np.ndarray]:
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


def state_to_tensors(state: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix, scalar = state_to_arrays(state)
    return (
        torch.as_tensor(matrix, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(scalar, dtype=torch.float32, device=device).unsqueeze(0),
    )


def normalized_to_env_action(action: np.ndarray, action_scale: int) -> np.ndarray:
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    scaled = np.rint(clipped * action_scale)
    return np.clip(scaled, -action_scale, action_scale).astype(np.float32)


def max_decisions(mode: str, max_days: int) -> int:
    return 1 if mode == "long" else max_days


def step_env(env: WarehouseEnv, action: np.ndarray, mode: str, decision_index: int):
    if mode == "long":
        return env.step(action, first_step=True, pattern="long")
    if mode == "hybrid":
        return env.step(action, first_step=(decision_index == 0))
    return env.step(action)


def make_episode_env(base_env: WarehouseEnv, orders):
    env = copy.deepcopy(base_env)
    state = env.reset(copy.deepcopy(orders))
    return env, state


def collect_metrics(env: WarehouseEnv, episode: int, scenario: int, mode: str, seed: int) -> Dict:
    delay_cost = sum(order.total_delay_cost(env.current_time) for order in env.orders_arrived)
    robot_cost = sum(robot.total_run_cost(env.current_time) for robot in env.robots_added)
    picker_cost = sum(picker.total_hire_cost(env.current_time) for picker in env.pickers_added)
    total_cost = delay_cost + robot_cost + picker_cost

    completed_orders = env.orders_completed
    completed_count = len(completed_orders)
    total_orders = len(env.orders_arrived)
    on_time = len([order for order in completed_orders if order.complete_time <= order.due_time])
    avg_picking_time = 0.0
    if completed_count > 0:
        avg_picking_time = sum(
            order.complete_time - order.arrive_time for order in completed_orders
        ) / completed_count
    completion_rate = on_time / total_orders if total_orders > 0 else 0.0

    return {
        "episode": episode,
        "total_cost": total_cost,
        "delay_cost": delay_cost,
        "robot_cost": robot_cost,
        "picker_cost": picker_cost,
        "completed_orders": completed_count,
        "on_time_completed_orders": on_time,
        "total_orders": total_orders,
        "average_picking_time": avg_picking_time,
        "completion_rate": completion_rate,
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
    }


def metrics_row(metrics: Dict) -> List:
    return [metrics[key] for key in CSV_HEADER]


class CsvLogger:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(CSV_HEADER)

    def write(self, metrics: Dict) -> None:
        self.writer.writerow(metrics_row(metrics))
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class NullViz:
    def line(self, *args, **kwargs):
        return None


def make_visdom(args: argparse.Namespace, algorithm: str):
    if not args.visdom:
        return NullViz(), f"{algorithm}_{args.mode}"
    try:
        from visdom import Visdom

        viz = Visdom(env="DRL_Baselines")
        win = f"{algorithm}_{args.mode}_s{args.scenario}_seed{args.seed}"
        viz.line(
            [0],
            [0],
            win=win,
            opts=dict(title=win, xlabel="Episode", ylabel="Total Cost"),
        )
        return viz, win
    except Exception as exc:  # Visdom should never stop training.
        print(f"Visdom disabled: {exc}")
        return NullViz(), f"{algorithm}_{args.mode}"


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


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


def save_checkpoint(path: Path, modules: Dict[str, nn.Module], extra: Dict | None = None) -> None:
    checkpoint = {name: module.state_dict() for name, module in modules.items()}
    if extra:
        checkpoint["extra"] = extra
    torch.save(checkpoint, path)


def print_episode(algorithm: str, metrics: Dict) -> None:
    print(
        f"{algorithm.upper()} episode {metrics['episode']}: "
        f"cost={metrics['total_cost']:.2f}, "
        f"orders={metrics['completed_orders']}/{metrics['total_orders']}, "
        f"rate={metrics['completion_rate']:.4f}"
    )
