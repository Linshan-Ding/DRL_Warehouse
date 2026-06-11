from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from agent.training_utils import (
    CsvLogger,
    best_config_path,
    case_stem,
    close_loggers,
    collect_metrics,
    collect_resource_config,
    episode_month,
    episode_month_counts,
    format_month_counts,
    init_base_env,
    initialize_episode_resources,
    load_orders_by_month,
    make_episode_env,
    make_monthly_loggers,
    make_training_visdom,
    max_decisions,
    monthly_best_config_path,
    monthly_metrics_path,
    order_instance_path,
    output_paths,
    polling_training_enabled,
    set_seed,
    step_env,
    training_months,
    write_best_config_csv,
)
from environment.class_public import load_config, print_config_source
from environment.warehouse_env import WarehouseEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train unified PPO agents for DRL_Warehouse.")
    return parser


def make_networks(env: WarehouseEnv, parameters: dict[str, Any]):
    from agent.ppo.networks import PolicyNetwork, ValueNetwork

    ppo = parameters["ppo"]
    scalar_dim = env.N_a + 1
    action_dim = env.N_a + 1
    policy = PolicyNetwork(
        input_height=env.N_w,
        input_width=env.N_l,
        scalar_dim=scalar_dim,
        output_dim=action_dim,
        hidden_dim=ppo["hidden_dim"],
        feature_dim=ppo["feature_dim"],
        attention_heads=ppo["attention_heads"],
        initial_log_std=ppo["initial_log_std"],
        min_log_std=ppo["min_log_std"],
        max_log_std=ppo["max_log_std"],
    )
    value = ValueNetwork(
        input_height=env.N_w,
        input_width=env.N_l,
        scalar_dim=scalar_dim,
        hidden_dim=ppo["hidden_dim"],
        feature_dim=ppo["feature_dim"],
    )
    return policy, value


def print_training_parameters(
    parameters: dict[str, Any],
    env: WarehouseEnv,
    decision_limit: int,
    order_path: Path | dict[int, Path],
    csv_path: Path,
    model_path: Path,
    best_config_csv_path: Path,
    polling_months: list[int] | None = None,
    monthly_csv_paths: dict[int, Path] | None = None,
    monthly_best_config_paths: dict[int, Path] | None = None,
) -> None:
    experiment = parameters["experiment"]
    ppo = parameters["ppo"]
    warehouse = parameters["warehouse"]
    months = polling_months or training_months(parameters)
    month_counts = episode_month_counts(months, int(experiment["episodes"]))

    lines = [
        "Training parameters:",
        f"  experiment.mode: {experiment['mode']}",
        f"  experiment.item_scenario: {experiment['item_scenario']}",
        f"  experiment.month: {experiment['month']}",
        f"  experiment.polling_training_enabled: {experiment['polling_training_enabled']}",
        f"  experiment.polling_months: {months}",
        f"  polling.month_episode_counts: {format_month_counts(month_counts)}",
        f"  experiment.episodes: {experiment['episodes']}",
        f"  experiment.seed: {experiment['seed']}",
        f"  experiment.max_days: {experiment['max_days']}",
        f"  experiment.decision_limit: {decision_limit}",
        f"  experiment.total_days: {experiment['total_days']}",
        f"  experiment.work_seconds_per_day: {experiment['work_seconds_per_day']}",
        f"  experiment.device: {experiment['device']}",
        f"  experiment.visdom: {experiment['visdom']}",
    ]
    if experiment["mode"] == "fixed_hybrid":
        fixed_hybrid = experiment["fixed_hybrid"]
        lines.extend(
            [
                f"  fixed_hybrid.long_term_robots: {fixed_hybrid['long_term_robots']}",
                f"  fixed_hybrid.long_term_pickers_area: {fixed_hybrid['long_term_pickers_area']}",
            ]
        )

    if isinstance(order_path, dict):
        order_lines = [
            f"  paths.orders.m{month:02d}: {path}"
            for month, path in order_path.items()
        ]
    else:
        order_lines = [f"  paths.orders: {order_path}"]
    lines.extend(order_lines)
    lines.extend(
        [
            f"  paths.ppo_csv: {csv_path}",
            f"  paths.ppo_model: {model_path}",
            f"  paths.best_config_csv: {best_config_csv_path}",
        ]
    )
    if monthly_csv_paths:
        lines.extend(
            f"  paths.monthly_csv.m{month:02d}: {path}"
            for month, path in monthly_csv_paths.items()
        )
    if monthly_best_config_paths:
        lines.extend(
            f"  paths.monthly_best_config.m{month:02d}: {path}"
            for month, path in monthly_best_config_paths.items()
        )
    lines.extend(
        [
            f"  ppo.learning_rate: {ppo['learning_rate']}",
            f"  ppo.gamma: {ppo['gamma']}",
            f"  ppo.batch_size: {ppo['batch_size']}",
            f"  ppo.n_epochs: {ppo['n_epochs']}",
            f"  ppo.gae_lambda: {ppo['gae_lambda']}",
            f"  ppo.clip_range: {ppo['clip_range']}",
            f"  ppo.initial_entropy_coeff: {ppo['initial_entropy_coeff']}",
            f"  ppo.min_entropy_coeff: {ppo['min_entropy_coeff']}",
            f"  ppo.entropy_coeff_decay: {ppo['entropy_coeff_decay']}",
            f"  ppo.hidden_dim: {ppo['hidden_dim']}",
            f"  ppo.feature_dim: {ppo['feature_dim']}",
            f"  ppo.attention_heads: {ppo['attention_heads']}",
            f"  ppo.initial_log_std: {ppo['initial_log_std']}",
            f"  ppo.min_log_std: {ppo['min_log_std']}",
            f"  ppo.max_log_std: {ppo['max_log_std']}",
            f"  warehouse.area_num: {warehouse['area_num']}",
            f"  warehouse.aisle_num: {warehouse['aisle_num']}",
            f"  warehouse.shelf_capacity: {warehouse['shelf_capacity']}",
            f"  warehouse.shelf_levels: {warehouse['shelf_levels']}",
            f"  env.total_time: {env.total_time}",
        ]
    )
    print("\n".join(lines))


def _done_for_update(mode: str, done: bool) -> bool:
    return True if mode == "long" else done


def _should_update(
    mode: str,
    buffer_len: int,
    batch_size: int,
    episode: int,
    total_episodes: int,
) -> bool:
    if mode != "long":
        return True
    return buffer_len >= batch_size


def _format_update_diagnostics(episode: int, stats: dict[str, float]) -> str:
    return (
        f"PPO update episode {episode}: "
        f"mean_reward={stats['mean_reward']:.4f}, "
        f"std_reward={stats['std_reward']:.4f}, "
        f"mean_return={stats['mean_return']:.4f}, "
        f"policy_loss={stats['policy_loss']:.4f}, "
        f"value_loss={stats['value_loss']:.4f}, "
        f"entropy={stats['entropy']:.4f}, "
        f"mean_action_std={stats['mean_action_std']:.4f}"
    )


def train(parameters: dict[str, Any]) -> Path:
    import torch

    from agent.ppo.agent import PPOAgent

    experiment = parameters["experiment"]
    mode = experiment["mode"]
    item_count = int(experiment["item_scenario"])
    month = int(experiment["month"])
    months = training_months(parameters)
    polling_enabled = polling_training_enabled(parameters)
    episodes = int(experiment["episodes"])
    seed = int(experiment["seed"])
    decision_limit = max_decisions(mode, int(experiment["max_days"]))
    set_seed(seed)

    base_env = init_base_env(parameters)
    policy, value = make_networks(base_env, parameters)
    agent = PPOAgent(policy, value, parameters=parameters, device=experiment["device"])
    stem = case_stem(
        mode,
        item_count,
        month,
        seed,
        polling_months=months if polling_enabled else None,
    )
    csv_path, model_path = output_paths(
        parameters["paths"]["ppo_output_dir"],
        stem,
    )
    best_config_csv_path = best_config_path(parameters["paths"]["ppo_output_dir"], stem)
    order_paths = {
        current_month: order_instance_path(parameters, item_count, current_month)
        for current_month in months
    }
    monthly_csv_paths = (
        {
            current_month: monthly_metrics_path(
                parameters["paths"]["ppo_output_dir"],
                stem,
                current_month,
            )
            for current_month in months
        }
        if polling_enabled
        else {}
    )
    monthly_best_config_paths = (
        {
            current_month: monthly_best_config_path(
                parameters["paths"]["ppo_output_dir"],
                stem,
                current_month,
            )
            for current_month in months
        }
        if polling_enabled
        else {}
    )
    print_training_parameters(
        parameters,
        base_env,
        decision_limit,
        order_paths if polling_enabled else order_paths[month],
        csv_path,
        model_path,
        best_config_csv_path,
        polling_months=months,
        monthly_csv_paths=monthly_csv_paths,
        monthly_best_config_paths=monthly_best_config_paths,
    )
    orders_by_month = load_orders_by_month(parameters, item_count, months)
    viz = make_training_visdom(
        bool(experiment["visdom"]),
        "DRL_PPO",
        f"ppo_{stem}",
        f"PPO {stem}",
        polling_enabled=polling_enabled,
        months=months,
    )

    best_cost = float("inf")
    monthly_best_cost = {current_month: float("inf") for current_month in months}
    if mode == "long":
        agent.buffer.clear()
    logger = CsvLogger(csv_path)
    monthly_loggers = make_monthly_loggers(
        parameters["paths"]["ppo_output_dir"],
        stem,
        months,
    ) if polling_enabled else {}
    try:
        for episode in range(1, episodes + 1):
            current_month = episode_month(months, episode)
            orders = orders_by_month[current_month]
            env, state = make_episode_env(base_env, orders)
            if mode != "long":
                agent.buffer.clear()
            state, done, first_decision_index, episode_config_rows = initialize_episode_resources(
                env,
                parameters,
                mode,
                episode,
                item_count,
                current_month,
                seed,
                "ppo",
            )
            learned_steps = 0

            for decision in range(first_decision_index, decision_limit):
                if done:
                    break
                action, log_prob, value_estimate, matrix_state, scalar_state = agent.select_action(state)
                decision_start_time = env.current_time
                next_state, reward, done = step_env(env, action, mode, decision)
                learned_steps += 1
                agent.buffer.add(
                    matrix_state,
                    scalar_state,
                    action,
                    log_prob,
                    reward,
                    _done_for_update(mode, done),
                    value_estimate,
                    maxlen=agent.batch_size if mode == "long" else None,
                )
                decision_metrics = collect_metrics(env, episode, item_count, current_month, mode, seed)
                episode_config_rows.append(
                    collect_resource_config(
                        env,
                        episode,
                        decision,
                        action,
                        decision_metrics,
                        "ppo",
                        item_count,
                        current_month,
                        mode,
                        seed,
                        decision_start_time=decision_start_time,
                    )
                )
                state = next_state

            updated = False
            if learned_steps > 0 and _should_update(mode, len(agent.buffer), agent.batch_size, episode, episodes):
                updated = agent.update(clear_buffer=(mode != "long"))
            metrics = collect_metrics(env, episode, item_count, current_month, mode, seed)
            logger.write(metrics)
            if current_month in monthly_loggers:
                monthly_loggers[current_month].write(metrics)
            viz.line_total_cost(metrics)
            if updated and agent.last_update_stats:
                print(_format_update_diagnostics(episode, agent.last_update_stats))

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
            if (
                current_month in monthly_best_config_paths
                and metrics["total_cost"] < monthly_best_cost[current_month]
            ):
                monthly_best_cost[current_month] = metrics["total_cost"]
                write_best_config_csv(monthly_best_config_paths[current_month], episode_config_rows)

            print(
                f"PPO {mode} episode {episode}: "
                f"cost={metrics['total_cost']:.2f}, "
                f"orders={metrics['completed_orders']}/{metrics['total_orders']}, "
                f"rate={metrics['completion_rate']:.4f}"
            )
    finally:
        logger.close()
        close_loggers(monthly_loggers)

    return csv_path


def main() -> None:
    build_parser().parse_args()
    print_config_source()
    parameters = load_config()
    csv_path = train(parameters)
    print(f"Saved PPO metrics to {csv_path}")


if __name__ == "__main__":
    main()
