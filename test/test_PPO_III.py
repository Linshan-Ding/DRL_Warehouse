from __future__ import annotations

import importlib.util
import argparse
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class PPOCliTest(unittest.TestCase):
    def test_parser_accepts_public_interface(self):
        from agent.ppo.train import build_parser

        args = build_parser().parse_args(
            ["--mode", "short", "--items", "2", "--month", "1", "--episodes", "1", "--seed", "0"]
        )
        self.assertEqual(args.mode, "short")
        self.assertEqual(args.items, 2)
        self.assertEqual(args.month, 1)
        self.assertEqual(args.episodes, 1)
        self.assertEqual(args.seed, 0)

    def test_baseline_parser_accepts_item_and_month(self):
        from agent.baselines.common import add_common_args

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args(
            ["--mode", "short", "--items", "2", "--month", "1", "--episodes", "1", "--seed", "0"]
        )
        self.assertEqual(args.items, 2)
        self.assertEqual(args.month, 1)


if __name__ == "__main__":
    unittest.main()
