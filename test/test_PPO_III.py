from __future__ import annotations

import importlib.util
import argparse
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path


class RolloutBufferTest(unittest.TestCase):
    def test_trim_to_maxlen_keeps_recent_transitions_and_aligned_fields(self):
        from agent.ppo.buffer import RolloutBuffer

        buffer = RolloutBuffer()
        for value in range(5):
            buffer.add(value, value + 10, value + 20, value + 30, value + 40, value + 50, value + 60, maxlen=3)

        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.matrix_states, [2, 3, 4])
        self.assertEqual(buffer.scalar_states, [12, 13, 14])
        self.assertEqual(buffer.actions, [22, 23, 24])
        self.assertEqual(buffer.logprobs, [32, 33, 34])
        self.assertEqual(buffer.rewards, [42, 43, 44])
        self.assertEqual(buffer.dones, [52, 53, 54])
        self.assertEqual(buffer.values, [62, 63, 64])


class CliConvergenceTest(unittest.TestCase):
    def test_generate_orders_parser_rejects_default_config_overrides(self):
        from data.generate_orders import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--output-dir", "data/instances"])


class PollingTrainingHelperTest(unittest.TestCase):
    class _FakeViz:
        def __init__(self):
            self.calls = []

        def line(self, y, x, **kwargs):
            self.calls.append((list(y), list(x), kwargs))

    def test_training_months_uses_single_month_when_polling_disabled(self):
        from agent.training_utils import training_months
        from environment.class_public import Config

        parameters = copy.deepcopy(Config().parameters)
        parameters["experiment"]["polling_training_enabled"] = False
        parameters["experiment"]["month"] = 9
        parameters["experiment"]["months"] = [1, 2, 3]

        self.assertEqual(training_months(parameters), [9])

    def test_training_months_uses_month_list_when_polling_enabled(self):
        from agent.training_utils import training_months
        from environment.class_public import Config

        parameters = copy.deepcopy(Config().parameters)
        parameters["experiment"]["polling_training_enabled"] = True
        parameters["experiment"]["months"] = [1, 2, 3]

        self.assertEqual(training_months(parameters), [1, 2, 3])

    def test_episode_month_cycles_through_months(self):
        from agent.training_utils import episode_month

        months = [1, 2, 3]
        mapped = [episode_month(months, episode) for episode in range(1, 6)]

        self.assertEqual(mapped, [1, 2, 3, 1, 2])

    def test_episode_month_counts_assigns_ordered_remainder(self):
        from agent.training_utils import episode_month_counts

        self.assertEqual(episode_month_counts([1, 2, 3], 5), {1: 2, 2: 2, 3: 1})

    def test_load_orders_by_month_reports_missing_paths(self):
        from agent.training_utils import load_orders_by_month
        from environment.class_public import Config

        parameters = copy.deepcopy(Config().parameters)
        with tempfile.TemporaryDirectory() as temp_dir:
            parameters["paths"]["instance_dir"] = temp_dir

            with self.assertRaises(FileNotFoundError) as ctx:
                load_orders_by_month(parameters, item_count=2, months=[1, 2])

        message = str(ctx.exception)
        self.assertIn("orders_m01.pkl", message)
        self.assertIn("orders_m02.pkl", message)

    def test_case_stem_marks_polling_months(self):
        from agent.training_utils import case_stem

        stem = case_stem("fixed_hybrid", 2, 9, 0, polling_months=[1, 2, 3])

        self.assertEqual(stem, "fixed_hybrid_i2_poll_m01-03_seed0")

    def test_disabled_monthly_visdom_has_stable_window_names(self):
        from agent.training_utils import make_training_visdom

        viz = make_training_visdom(
            False,
            "DRL_Test",
            "ppo_fixed_hybrid_i2_poll_m01-03_seed0",
            "PPO fixed_hybrid_i2_poll_m01-03_seed0",
            polling_enabled=True,
            months=[1, 2, 3],
        )

        self.assertEqual(
            viz.window_names(),
            {
                1: "ppo_fixed_hybrid_i2_poll_m01-03_seed0_m01",
                2: "ppo_fixed_hybrid_i2_poll_m01-03_seed0_m02",
                3: "ppo_fixed_hybrid_i2_poll_m01-03_seed0_m03",
            },
        )
        viz.line_total_cost({"episode": 1, "month": 1, "total_cost": 10})

    def test_monthly_visdom_routes_points_by_training_month(self):
        from agent.training_utils import TrainingViz

        fake_m01 = self._FakeViz()
        fake_m02 = self._FakeViz()
        fake_m03 = self._FakeViz()
        viz = TrainingViz(
            monthly={
                1: (fake_m01, "win_m01"),
                2: (fake_m02, "win_m02"),
                3: (fake_m03, "win_m03"),
            }
        )

        for episode, month in enumerate([1, 2, 3, 1, 2], start=1):
            viz.line_total_cost(
                {"episode": episode, "month": month, "total_cost": episode * 10}
            )

        self.assertEqual([call[1][0] for call in fake_m01.calls], [1, 4])
        self.assertEqual([call[1][0] for call in fake_m02.calls], [2, 5])
        self.assertEqual([call[1][0] for call in fake_m03.calls], [3])
        self.assertTrue(all(call[2]["update"] == "append" for call in fake_m01.calls))

    def test_single_visdom_routes_all_points_to_default_window(self):
        from agent.training_utils import TrainingViz

        fake = self._FakeViz()
        viz = TrainingViz(default=(fake, "single_window"))

        for episode in range(1, 4):
            viz.line_total_cost({"episode": episode, "month": 9, "total_cost": episode})

        self.assertEqual([call[1][0] for call in fake.calls], [1, 2, 3])
        self.assertTrue(all(call[2]["win"] == "single_window" for call in fake.calls))


class PPOTrainingSummaryTest(unittest.TestCase):
    def test_training_parameter_summary_includes_core_values(self):
        from agent.ppo.train import print_training_parameters
        from environment.class_public import Config
        from environment.warehouse_env import WarehouseEnv

        parameters = copy.deepcopy(Config().parameters)
        env = WarehouseEnv()
        env.total_time = 123
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_training_parameters(
                parameters,
                env,
                decision_limit=31,
                order_path=Path("data/instances/items_2/orders_m09.pkl"),
                csv_path=Path("result/ppo/fixed_hybrid_i2_m09_seed0.csv"),
                model_path=Path("result/ppo/fixed_hybrid_i2_m09_seed0.pth"),
                best_config_csv_path=Path("result/ppo/fixed_hybrid_i2_m09_seed0_best_config.csv"),
            )

        text = output.getvalue()
        self.assertIn("Training parameters:", text)
        self.assertIn("experiment.mode:", text)
        self.assertIn("experiment.item_scenario:", text)
        self.assertIn("experiment.polling_training_enabled:", text)
        self.assertIn("ppo.learning_rate:", text)
        self.assertIn("paths.ppo_csv:", text)

    def test_training_parameter_summary_includes_fixed_hybrid_values(self):
        from agent.ppo.train import print_training_parameters
        from environment.class_public import Config
        from environment.warehouse_env import WarehouseEnv

        parameters = copy.deepcopy(Config().parameters)
        parameters["experiment"]["mode"] = "fixed_hybrid"
        env = WarehouseEnv()
        env.total_time = 123
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_training_parameters(
                parameters,
                env,
                decision_limit=31,
                order_path=Path("orders.pkl"),
                csv_path=Path("metrics.csv"),
                model_path=Path("model.pth"),
                best_config_csv_path=Path("best.csv"),
            )

        text = output.getvalue()
        self.assertIn("fixed_hybrid.long_term_robots:", text)
        self.assertIn("fixed_hybrid.long_term_pickers_area:", text)

    def test_training_parameter_summary_includes_polling_values(self):
        from agent.ppo.train import print_training_parameters
        from environment.class_public import Config
        from environment.warehouse_env import WarehouseEnv

        parameters = copy.deepcopy(Config().parameters)
        parameters["experiment"]["polling_training_enabled"] = True
        parameters["experiment"]["months"] = [1, 2, 3]
        parameters["experiment"]["episodes"] = 5
        env = WarehouseEnv()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_training_parameters(
                parameters,
                env,
                decision_limit=31,
                order_path={
                    1: Path("data/instances/items_2/orders_m01.pkl"),
                    2: Path("data/instances/items_2/orders_m02.pkl"),
                    3: Path("data/instances/items_2/orders_m03.pkl"),
                },
                csv_path=Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0.csv"),
                model_path=Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0.pth"),
                best_config_csv_path=Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0_best_config.csv"),
                polling_months=[1, 2, 3],
                monthly_csv_paths={
                    1: Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0_m01.csv"),
                    2: Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0_m02.csv"),
                    3: Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0_m03.csv"),
                },
                monthly_best_config_paths={
                    1: Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0_m01_best_config.csv"),
                    2: Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0_m02_best_config.csv"),
                    3: Path("result/ppo/fixed_hybrid_i2_poll_m01-03_seed0_m03_best_config.csv"),
                },
            )

        text = output.getvalue()
        self.assertIn("experiment.polling_training_enabled: True", text)
        self.assertIn("experiment.polling_months: [1, 2, 3]", text)
        self.assertIn("polling.month_episode_counts: m01=2, m02=2, m03=1", text)
        self.assertIn("paths.monthly_csv.m01:", text)
        self.assertIn("paths.monthly_best_config.m01:", text)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class PPOCliTest(unittest.TestCase):
    def _make_test_agent(self):
        import copy

        from agent.ppo.agent import PPOAgent
        from agent.ppo.train import make_networks
        from environment.class_public import Config
        from environment.warehouse_env import WarehouseEnv

        parameters = copy.deepcopy(Config().parameters)
        parameters["experiment"]["device"] = "cpu"
        env = WarehouseEnv()
        policy, value = make_networks(env, parameters)
        return PPOAgent(policy, value, parameters=parameters, device="cpu"), env, parameters

    def test_ppo_parser_rejects_default_config_overrides(self):
        from agent.ppo.train import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--mode", "fixed_hybrid"])

    def test_baseline_parser_rejects_default_config_overrides(self):
        from agent.baselines.common import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--seed", "0"])

    def test_baseline_parser_uses_default_config_values(self):
        from agent.baselines.common import add_common_args
        from environment.class_public import Config

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        experiment = Config().parameters["experiment"]

        self.assertEqual(args.mode, experiment["mode"])
        self.assertEqual(args.items, experiment["item_scenario"])
        self.assertEqual(args.month, experiment["month"])
        self.assertEqual(args.seed, experiment["seed"])

    def test_long_mode_done_for_update_is_one_step(self):
        from agent.ppo.train import _done_for_update

        self.assertTrue(_done_for_update("long", False))
        self.assertTrue(_done_for_update("long", True))

    def test_short_and_hybrid_done_for_update_preserve_environment_done(self):
        from agent.ppo.train import _done_for_update

        self.assertFalse(_done_for_update("short", False))
        self.assertTrue(_done_for_update("short", True))
        self.assertFalse(_done_for_update("hybrid", False))
        self.assertTrue(_done_for_update("hybrid", True))
        self.assertFalse(_done_for_update("fixed_hybrid", False))
        self.assertTrue(_done_for_update("fixed_hybrid", True))

    def test_long_mode_update_waits_for_full_sliding_window(self):
        from agent.ppo.train import _should_update

        self.assertFalse(_should_update("long", buffer_len=1, batch_size=64, episode=1, total_episodes=1))
        self.assertFalse(_should_update("long", buffer_len=3, batch_size=64, episode=3, total_episodes=10))
        self.assertTrue(_should_update("long", buffer_len=64, batch_size=64, episode=4, total_episodes=10))
        self.assertFalse(_should_update("long", buffer_len=3, batch_size=64, episode=10, total_episodes=10))

    def test_short_and_hybrid_update_every_episode(self):
        from agent.ppo.train import _should_update

        self.assertTrue(_should_update("short", buffer_len=0, batch_size=64, episode=1, total_episodes=10))
        self.assertTrue(_should_update("hybrid", buffer_len=0, batch_size=64, episode=1, total_episodes=10))
        self.assertTrue(_should_update("fixed_hybrid", buffer_len=0, batch_size=64, episode=1, total_episodes=10))

    def test_reward_standardization_centers_and_scales_batch(self):
        import torch

        agent, _, _ = self._make_test_agent()
        rewards = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=agent.device)
        processed = agent._preprocess_rewards(rewards)

        self.assertAlmostEqual(float(processed.mean().cpu()), 0.0, places=6)
        self.assertAlmostEqual(float(processed.std(unbiased=False).cpu()), 1.0, places=6)

    def test_reward_standardization_handles_low_variance_batch(self):
        import torch

        agent, _, _ = self._make_test_agent()
        rewards = torch.as_tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=agent.device)
        processed = agent._preprocess_rewards(rewards)

        self.assertTrue(torch.isfinite(processed).all().item())
        self.assertAlmostEqual(float(processed.abs().max().cpu()), 0.0, places=6)

    def test_state_normalization_preserves_shapes_and_expected_scales(self):
        import numpy as np
        import torch

        agent, env, parameters = self._make_test_agent()
        warehouse = parameters["warehouse"]
        max_order_items = max(parameters["order"]["order_n_items"])
        pick_points_total = warehouse["area_num"] * warehouse["aisle_num"] * warehouse["shelf_capacity"]
        pick_points_per_area = warehouse["aisle_num"] * warehouse["shelf_capacity"]
        shape = (env.N_w, env.N_l)
        state = {
            "robot_queue_list": np.full(shape, pick_points_total, dtype=np.float32),
            "picker_list": np.ones(shape, dtype=np.float32),
            "unpicked_items_list": np.full(shape, max_order_items, dtype=np.float32),
            "n_robots": pick_points_total,
            "n_pickers_area": [pick_points_per_area] * env.N_a,
        }

        matrix, scalar = agent._state_to_tensors(state)

        self.assertEqual(tuple(matrix.shape), (1, 3, env.N_w, env.N_l))
        self.assertEqual(tuple(scalar.shape), (1, env.N_a + 1))
        self.assertTrue(torch.isfinite(matrix).all().item())
        self.assertTrue(torch.isfinite(scalar).all().item())
        self.assertAlmostEqual(float(matrix.max().cpu()), 1.0, places=6)
        self.assertAlmostEqual(float(scalar.max().cpu()), 1.0, places=6)

    def test_policy_action_log_std_is_clamped(self):
        import torch

        from agent.ppo.networks import PolicyNetwork

        policy = PolicyNetwork(
            input_height=2,
            input_width=2,
            scalar_dim=4,
            output_dim=4,
            initial_log_std=5.0,
            min_log_std=-0.25,
            max_log_std=0.25,
        )
        self.assertLessEqual(float(policy.action_log_std.max()), 0.25)

        with torch.no_grad():
            policy.action_log_std.fill_(2.0)
        policy.clamp_action_log_std_()
        self.assertLessEqual(float(policy.action_log_std.max()), 0.25)

        matrix = torch.zeros(1, 3, 2, 2)
        scalar = torch.zeros(1, 4)
        _, std = policy(matrix, scalar)
        self.assertLessEqual(float(std.max()), float(torch.exp(torch.tensor(0.25))) + 1e-6)

    def test_update_can_preserve_or_clear_buffer(self):
        import numpy as np

        agent, env, _ = self._make_test_agent()
        shape = (env.N_w, env.N_l)
        state = {
            "robot_queue_list": np.zeros(shape, dtype=np.float32),
            "picker_list": np.zeros(shape, dtype=np.float32),
            "unpicked_items_list": np.zeros(shape, dtype=np.float32),
            "n_robots": 1,
            "n_pickers_area": [1] * env.N_a,
        }
        for reward in (1.0, 2.0):
            action, log_prob, value, matrix_state, scalar_state = agent.select_action(state)
            agent.buffer.add(matrix_state, scalar_state, action, log_prob, reward, True, value)

        self.assertTrue(agent.update(clear_buffer=False))
        self.assertEqual(len(agent.buffer), 2)
        self.assertTrue(agent.update(clear_buffer=True))
        self.assertEqual(len(agent.buffer), 0)


if __name__ == "__main__":
    unittest.main()
