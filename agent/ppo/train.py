from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

import torch

from agent.ppo.agent import PPOAgent
from agent.ppo.networks import PolicyNetwork, ValueNetwork
from agent.training_utils import (
    CSV_HEADER,
    MODES,
    best_config_path,
    case_stem,
    collect_metrics,
    collect_resource_config,
    init_base_env,
    load_orders,
    make_episode_env,
    make_visdom,
    max_decisions,
    output_paths,
    set_seed,
    step_env,
    write_best_config_csv,
)
from environment.class_public import load_config
from environment.warehouse_env import WarehouseEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train unified PPO agents for DRL_Warehouse.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=MODES, default='long')
    parser.add_argument("--items", type=int, default=2)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def apply_cli_overrides(parameters: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    experiment = parameters["experiment"]
    paths = parameters["paths"]
    for name in ("mode", "episodes", "seed", "max_days", "device"):
        value = getattr(args, name)
        if value is not None:
            experiment[name] = value
    if args.items is not None:
        experiment["item_scenario"] = args.items
    if args.month is not None:
        experiment["month"] = args.month
    if args.output_dir is not None:
        paths["ppo_output_dir"] = args.output_dir
    if args.visdom:
        experiment["visdom"] = True
    return parameters


def make_networks(env: WarehouseEnv, parameters: dict[str, Any]):
    ppo = parameters["ppo"]
    scalar_dim = env.N_a + 1
    action_dim = env.N_a + 1
    policy = PolicyNetwork(
        input_height=env.N_w,
        input_width=env.N_l,
        scalar_dim=scalar_dim,
        output_dim=action_dim,
        hidden_dim=ppo.get("hidden_dim", 128),
        feature_dim=ppo.get("feature_dim", 32),
        attention_heads=ppo.get("attention_heads", 4),
        initial_log_std=ppo.get("initial_log_std", 1.5),
    )
    value = ValueNetwork(
        input_height=env.N_w,
        input_width=env.N_l,
        scalar_dim=scalar_dim,
        hidden_dim=ppo.get("hidden_dim", 128),
        feature_dim=ppo.get("feature_dim", 32),
    )
    return policy, value


def train(parameters: dict[str, Any]) -> Path:
    experiment = parameters["experiment"]
    mode = experiment["mode"]
    item_count = int(experiment["item_scenario"])
    month = int(experiment["month"])
    episodes = int(experiment["episodes"])
    seed = int(experiment["seed"])
    set_seed(seed)

    base_env = init_base_env(parameters)
    orders = load_orders(parameters, item_count, month)
    policy, value = make_networks(base_env, parameters)
    agent = PPOAgent(policy, value, parameters=parameters, device=experiment.get("device"))
    stem = case_stem(mode, item_count, month, seed)
    csv_path, model_path = output_paths(
        parameters["paths"]["ppo_output_dir"],
        stem,
    )
    best_config_csv_path = best_config_path(parameters["paths"]["ppo_output_dir"], stem)
    viz, viz_win = make_visdom(
        bool(experiment.get("visdom", False)),
        "DRL_PPO",
        f"ppo_{stem}",
    )

    best_cost = float("inf")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for episode in range(1, episodes + 1):
            env, state = make_episode_env(base_env, orders)
            agent.buffer.clear()
            done = False
            episode_config_rows = []

            for decision in range(max_decisions(mode, int(experiment["max_days"]))):
                if done:
                    break
                action, log_prob, value_estimate, matrix_state, scalar_state = agent.select_action(state)
                decision_start_time = env.current_time
                next_state, reward, done = step_env(env, action, mode, decision)
                agent.buffer.add(matrix_state, scalar_state, action, log_prob, reward, done, value_estimate)
                decision_metrics = collect_metrics(env, episode, item_count, month, mode, seed)
                episode_config_rows.append(
                    collect_resource_config(
                        env,
                        episode,
                        decision,
                        action,
                        decision_metrics,
                        "ppo",
                        item_count,
                        month,
                        mode,
                        seed,
                        decision_start_time=decision_start_time,
                    )
                )
                state = next_state

            agent.update()
            metrics = collect_metrics(env, episode, item_count, month, mode, seed)
            writer.writerow([metrics[key] for key in CSV_HEADER])
            f.flush()
            viz.line([metrics["total_cost"]], [episode], win=viz_win, update="append")

            if metrics["total_cost"] < best_cost:
                best_cost = metrics["total_cost"]
                torch.save(
                    {
                        "policy": agent.policy.state_dict(),
                        "value": agent.value_network.state_dict(),
                        "config": parameters,
                        "metrics": metrics,
                    },
                    model_path,
                )
                write_best_config_csv(best_config_csv_path, episode_config_rows)

            print(
                f"PPO {mode} episode {episode}: "
                f"cost={metrics['total_cost']:.2f}, "
                f"orders={metrics['completed_orders']}/{metrics['total_orders']}, "
                f"rate={metrics['completion_rate']:.4f}"
            )

    return csv_path


def main() -> None:
    args = build_parser().parse_args()
    parameters = apply_cli_overrides(load_config(args.config), args)
    csv_path = train(parameters)
    print(f"Saved PPO metrics to {csv_path}")


if __name__ == "__main__":
    main()
