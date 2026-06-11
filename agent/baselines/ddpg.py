from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    CsvLogger,
    DeterministicActor,
    QNetwork,
    ReplayBuffer,
    add_common_args,
    best_config_path,
    close_loggers,
    collect_metrics,
    collect_resource_config,
    episode_month,
    hard_update,
    init_base_env,
    initialize_episode_resources,
    load_orders_by_month,
    make_episode_env,
    make_monthly_loggers,
    make_training_visdom,
    max_decisions,
    monthly_best_config_paths,
    normalized_to_env_action,
    output_paths,
    print_episode,
    print_config_source,
    save_checkpoint,
    set_seed,
    soft_update,
    state_to_arrays,
    state_to_tensors,
    step_env,
    training_months,
    write_best_config_csv,
)


def build_parser():
    parser = argparse.ArgumentParser(description="DDPG baseline for warehouse resource allocation.")
    add_common_args(parser)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    return parser


def update_ddpg(actor, actor_target, critic, critic_target, actor_optimizer, critic_optimizer, buffer, args, device):
    matrix, scalar, action, reward, next_matrix, next_scalar, done = buffer.sample(args.batch_size, device)
    with torch.no_grad():
        next_action = actor_target(next_matrix, next_scalar)
        target_q = critic_target(next_matrix, next_scalar, next_action)
        target = reward + args.gamma * (1 - done) * target_q

    critic_loss = nn.functional.mse_loss(critic(matrix, scalar, action), target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic(matrix, scalar, actor(matrix, scalar)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(actor, actor_target, args.tau)
    soft_update(critic, critic_target, args.tau)


def main():
    args = build_parser().parse_args()
    print_config_source()
    set_seed(args.seed)
    device = torch.device(args.device)

    months = training_months()
    orders_by_month = load_orders_by_month(args.items, months)
    base_env = init_base_env()
    decision_limit = max_decisions(args.mode, args.max_days)
    action_dim = base_env.N_a + 1
    scalar_dim = base_env.N_a + 1
    matrix_shape = (3, base_env.N_w, base_env.N_l)

    actor = DeterministicActor(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    actor_target = DeterministicActor(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    critic = QNetwork(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    critic_target = QNetwork(base_env.N_w, base_env.N_l, scalar_dim, action_dim).to(device)
    hard_update(actor, actor_target)
    hard_update(critic, critic_target)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    buffer = ReplayBuffer(args.buffer_size, matrix_shape, scalar_dim, action_dim)

    csv_path, model_path = output_paths(args, "ddpg", months)
    best_config_csv_path = best_config_path(args, "ddpg", months)
    monthly_best_paths = monthly_best_config_paths(args, "ddpg", months)
    logger = CsvLogger(csv_path)
    monthly_loggers = make_monthly_loggers(args, "ddpg", months)
    viz = make_training_visdom(args, "ddpg", months)
    best_cost = math.inf
    monthly_best_cost = {month: math.inf for month in months}
    global_step = 0

    try:
        for episode in range(1, args.episodes + 1):
            current_month = episode_month(months, episode)
            orders = orders_by_month[current_month]
            env, state = make_episode_env(base_env, orders)
            state, done, first_decision_index, episode_config_rows = initialize_episode_resources(
                env,
                args.mode,
                episode,
                args.items,
                current_month,
                args.seed,
                "ddpg",
            )
            for decision in range(first_decision_index, decision_limit):
                if done:
                    break
                matrix_np, scalar_np = state_to_arrays(state)
                if global_step < args.warmup_steps:
                    action_norm = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
                else:
                    matrix, scalar = state_to_tensors(state, device)
                    action_norm = actor(matrix, scalar).detach().cpu().numpy()[0]
                    noise = np.random.normal(0, args.exploration_noise, size=action_dim)
                    action_norm = np.clip(action_norm + noise, -1, 1).astype(np.float32)

                env_action = normalized_to_env_action(action_norm, args.action_scale)
                decision_start_time = env.current_time
                next_state, reward, done = step_env(env, env_action, args.mode, decision)
                terminal = done or decision == decision_limit - 1
                decision_metrics = collect_metrics(env, episode, args.items, current_month, args.mode, args.seed)
                episode_config_rows.append(
                    collect_resource_config(
                        env,
                        episode,
                        decision,
                        env_action,
                        decision_metrics,
                        "ddpg",
                        args.items,
                        current_month,
                        args.mode,
                        args.seed,
                        decision_start_time=decision_start_time,
                    )
                )
                next_matrix_np, next_scalar_np = state_to_arrays(next_state)
                buffer.add(matrix_np, scalar_np, action_norm, float(reward), next_matrix_np, next_scalar_np, terminal)

                if buffer.size >= args.batch_size:
                    update_ddpg(
                        actor,
                        actor_target,
                        critic,
                        critic_target,
                        actor_optimizer,
                        critic_optimizer,
                        buffer,
                        args,
                        device,
                    )

                state = next_state
                global_step += 1
                if terminal:
                    break

            metrics = collect_metrics(env, episode, args.items, current_month, args.mode, args.seed)
            logger.write(metrics)
            if current_month in monthly_loggers:
                monthly_loggers[current_month].write(metrics)
            viz.line_total_cost(metrics)
            print_episode("ddpg", metrics)

            if metrics["total_cost"] < best_cost:
                best_cost = metrics["total_cost"]
                save_checkpoint(model_path, {"actor": actor, "critic": critic}, vars(args))
                write_best_config_csv(best_config_csv_path, episode_config_rows)
            if current_month in monthly_best_paths and metrics["total_cost"] < monthly_best_cost[current_month]:
                monthly_best_cost[current_month] = metrics["total_cost"]
                write_best_config_csv(monthly_best_paths[current_month], episode_config_rows)
    finally:
        logger.close()
        close_loggers(monthly_loggers)


if __name__ == "__main__":
    main()
