from __future__ import annotations

import pickle
import unittest

from environment.class_public import Config, REPO_ROOT
from environment.warehouse_env import WarehouseEnv


class EnvironmentSmokeTest(unittest.TestCase):
    def test_items_2_month_1_reset_and_step(self):
        parameters = Config().parameters
        order_path = REPO_ROOT / parameters["paths"]["instance_dir"] / "items_2" / "orders_m01.pkl"
        self.assertTrue(order_path.exists(), f"Missing fixture: {order_path}")

        with order_path.open("rb") as f:
            orders = pickle.load(f)

        env = WarehouseEnv()
        env.total_time = (
            parameters["experiment"]["work_seconds_per_day"]
            * parameters["experiment"]["total_days"]
        )
        state = env.reset(orders)
        self.assertIn("robot_queue_list", state)

        action = [1, 1, 1, 1]
        next_state, reward, done = env.step(action, first_step=True)
        self.assertIn("n_robots", next_state)
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(done, bool)


if __name__ == "__main__":
    unittest.main()
