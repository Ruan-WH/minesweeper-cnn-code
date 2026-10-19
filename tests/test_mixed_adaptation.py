import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from minesweeper_train_mixed import (MixedConfig, MixedTrainer, load_reference, ROOT, file_hash,
                                    prepare_mixed_reference, configure_mixed_loss, validate_startup)


class TinyNet(nn.Module):
    """Cheap stochastic test network; NOT the paper network/checkpoint."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(810, 81))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, state):
        return self.fc(state)

    def compute_loss(self, prediction, label):
        return self.loss(prediction, label)


class FakeEnv:
    """Budget/accounting fixture; alternates visible outcomes, no renderer."""
    num_rows = 9
    num_cols = 9
    num_mines = 14
    unwrapped = property(lambda self: self)
    action_space = type("Actions", (), {"n": 81})()

    def reset(self):
        self.board = np.zeros((9, 9), dtype=np.int32)
        self.board.reshape(-1)[-self.num_mines:] = -2
        self.player_board = np.full((9, 9), -1, dtype=np.int32)
        self.reward, self.done, self.info = None, False, {}
        self.first_move, self.stats, self.button = True, {"n_win": 0}, "left"
        return self.observation()

    def observation(self):
        state = np.zeros((10, 9, 9), dtype=np.float32)
        state[9] = self.player_board == -1
        return state

    def step(self, action):
        assert self.player_board.reshape(-1)[action] == -1
        self.player_board.reshape(-1)[action] = 0
        self.reward = 1.0
        # Every third click ends the fixture game. Quota boundaries often cut it.
        self.done = np.count_nonzero(self.player_board == 0) == 3
        self.info = {"status": "win" if self.done else "progress"}
        self.stats["n_win"] = int(self.done)
        self.first_move = False
        return self.observation(), self.reward, self.done, self.info

    def close(self):
        pass


class MixedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        self.original_hash = file_hash(ROOT / "minesweeper_train_cnn.py")
        self.model_hash = file_hash(ROOT / "lib/models.py")

    def fixture_reference(self, directory):
        # Enable the original commented scheduler ONLY in a temporary test copy.
        # The actual project source and production configuration remain untouched.
        source = (ROOT / "minesweeper_train_cnn.py").read_text(encoding="utf-8")
        source = source.replace("#         self.scheduler =", "        self.scheduler =")
        source = source.replace("#         if hasattr(self, 'scheduler'):", "        if hasattr(self, 'scheduler'):")
        source = source.replace("#             self.scheduler.step()", "            self.scheduler.step()")
        path = Path(directory) / "reference_fixture.py"
        path.write_text(source, encoding="utf-8")
        return load_reference(path, directory)[:2]

    def make(self, directory, config, weights=None):
        reference, expressions = self.fixture_reference(directory)
        model = TinyNet()
        if weights is not None:
            model.load_state_dict(weights)
        return MixedTrainer(config, model, reference, expressions, directory, env=FakeEnv())

    def test_exact_budgets_shared_buffers_restart_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = MixedConfig(steps_14=73, steps_18=71, restart_at=70,
                                 buffer_size=32, epochs=1, window_size=5,
                                 checkpoint_every=1000, smoke=True)
            trainer = self.make(directory, config)
            features_id = id(trainer.agent.features)
            while trainer.total_steps < config.restart_at:
                trainer.step()
            self.assertEqual(trainer.restarts, [70])
            self.assertEqual(trainer.agent.current_sample, 0)
            self.assertEqual(len(trainer.agent.optimizer.state), 0)
            self.assertEqual(trainer.agent.scheduler.last_epoch, 0)
            self.assertEqual(trainer.agent.optimizer.param_groups[0]["lr"], trainer.reference.LEARNING_RATE)
            while trainer.step():
                pass
            self.assertEqual(trainer.steps, {14: 73, 18: 71})
            self.assertEqual(trainer.total_steps, 144)
            self.assertEqual(trainer.buffer_updates, {1: 2, 2: 2})
            self.assertEqual(trainer.adam_updates, {1: 2, 2: 2})
            self.assertEqual([r["samples"] for r in trainer.discarded], [6, 10])
            self.assertEqual(features_id, id(trainer.agent.features))
            self.assertGreater(trainer.mixed_buffers, 0)
            self.assertGreater(sum(trainer.truncated.values()), 0)
            for m in (14, 18):
                self.assertEqual(trainer.started[m], trainer.completed[m] + trainer.truncated[m])
            self.assertTrue((Path(directory) / "mixed_best_18_online.dat").is_file())
            self.assertTrue((Path(directory) / "mixed_final.dat").is_file())
            self.assertEqual(trainer.agent.scheduler.last_epoch, 2)
            self.assertEqual(self.original_hash, file_hash(ROOT / "minesweeper_train_cnn.py"))
            self.assertEqual(self.model_hash, file_hash(ROOT / "lib/models.py"))

    def test_resume_restores_midgame_partial_buffer_and_all_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            config = MixedConfig(steps_14=73, steps_18=71, restart_at=70,
                                 buffer_size=32, epochs=1, window_size=5,
                                 checkpoint_every=1000, smoke=True)
            trainer = self.make(directory, config)
            for _ in range(41):
                trainer.step()
            trainer.save_recovery()
            saved = torch.load(Path(directory) / "mixed_resume.pt", weights_only=False)
            self.assertEqual(saved["pending_samples"], 9)
            while trainer.step():
                pass
            expected = copy.deepcopy(trainer.model.state_dict())
            expected_summary = trainer.summary()
            # Reuse the same output; restore rolls back the checkpoint weights.
            restored = self.make(directory, config)
            restored.restore(saved)
            while restored.step():
                pass
            for name, tensor in expected.items():
                torch.testing.assert_close(tensor, restored.model.state_dict()[name], rtol=0, atol=0)
            for name in ("steps_14", "steps_18", "total_adaptation_steps", "games_completed", "wins",
                         "budget_truncated", "buffer_optimizations_by_phase", "adam_updates_by_phase",
                         "mixed_buffers", "discarded_samples", "best_18_online_win_rate"):
                self.assertEqual(expected_summary[name], restored.summary()[name])

    def test_actual_environment_labels_and_actions(self):
        from minesweeper_pygame import MinesweeperEnv
        from lib.wrappers import CustomObservationWrapper
        with tempfile.TemporaryDirectory() as directory:
            reference, expressions, _, _ = load_reference(ROOT / "minesweeper_train_cnn.py", directory)
            config = MixedConfig(steps_14=47, steps_18=43, restart_at=49,
                                 buffer_size=1024, epochs=10, window_size=5,
                                 checkpoint_every=1000, smoke=True)
            env = CustomObservationWrapper(MinesweeperEnv(9, 9, 14, "rgb_array"), "one_hot")
            trainer = MixedTrainer(config, TinyNet(), reference, expressions, directory, env=env)
            observed_mines = set()
            initial_inc = trainer.agent.increment_sample_size
            def checked(feature, label):
                observed_mines.add(trainer.current_mines)
                np.testing.assert_array_equal(label, (env.board == -2).reshape(-1))
                np.testing.assert_array_equal(feature, env.observation(env.unwrapped.player_board))
                initial_inc(feature, label)
            trainer.agent.increment_sample_size = checked
            while trainer.step():
                pass
            self.assertEqual(observed_mines, {14, 18})
            self.assertEqual(trainer.steps, {14: 47, 18: 43})
            self.assertEqual(trainer.adam_updates, {1: 0, 2: 0})
            self.assertEqual(sum(r["samples"] for r in trainer.discarded), 90)
            env.close()

    def test_paper_budget_arithmetic_is_not_an_experiment_result(self):
        config = MixedConfig()
        self.assertEqual(divmod(config.restart_at, config.buffer_size), (976, 576))
        self.assertEqual(divmod(config.steps_14 + config.steps_18 - config.restart_at,
                                config.buffer_size), (830, 80))
        self.assertEqual((976 + 830) * 10 * 32, 577920)

    def test_old_torch_step_counter_matches_native_training(self):
        config = MixedConfig(steps_14=73, steps_18=71, restart_at=70,
                             buffer_size=32, epochs=1, window_size=5,
                             checkpoint_every=1000, smoke=True)
        results = []
        for old_api in (False, True):
            random.seed(9)
            np.random.seed(9)
            torch.manual_seed(9)
            with tempfile.TemporaryDirectory() as directory:
                context = (mock.patch.object(torch.optim.Adam, "register_step_post_hook", None)
                           if old_api else mock.patch.dict({}, {}))
                with context:
                    trainer = self.make(directory, config)
                    while trainer.step():
                        pass
                    results.append((copy.deepcopy(trainer.model.state_dict()), trainer.summary()))
        for name, tensor in results[0][0].items():
            torch.testing.assert_close(tensor, results[1][0][name], rtol=0, atol=0)
        for field in ("adam_updates_by_phase", "buffer_optimizations_by_phase", "learning_rate",
                      "optimizer_restart_steps", "steps_14", "steps_18"):
            self.assertEqual(results[0][1][field], results[1][1][field])

    def test_original_augmentation_transforms_labels_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, _, _, _ = load_reference(ROOT / "minesweeper_train_cnn.py", directory)
            labels = np.arange(4 * 81).reshape(4, 81).astype(float)
            features = np.zeros((4, 10, 9, 9))
            features[:, 0] = labels.reshape(4, 9, 9)
            for seed in range(16):
                np.random.seed(seed)
                augmented_features, augmented_labels = reference.augment_data(features, labels)
                np.testing.assert_array_equal(augmented_features[:, 0].reshape(4, 81), augmented_labels)

    def test_real_model_pretrained_and_formal_startup_guards(self):
        from lib import models
        with tempfile.TemporaryDirectory() as directory:
            reference, expressions, _, augmentation = load_reference(ROOT / "minesweeper_train_cnn.py", directory)
            augmentation = prepare_mixed_reference(reference, augmentation)
            model = models.Model((10, 9, 9), 81)
            configure_mixed_loss(model)
            weights = torch.load(ROOT / "models/CNN-1763950674/Minesweeper-v0-best_win_rate.dat",
                                 map_location="cpu", weights_only=True)
            model.load_state_dict(weights, strict=True)
            trainer = MixedTrainer(MixedConfig(), model, reference, expressions, directory)
            startup = validate_startup(trainer, augmentation)
            self.assertEqual(startup["dropout"], [0.3, 0.3])
            self.assertEqual(startup["reduction"], "mean")
            self.assertIsNone(model.loss.weight)
            self.assertIsNone(model.loss.pos_weight)
            self.assertTrue(startup["augmentation"])
            scheduler = trainer.agent.scheduler
            del trainer.agent.scheduler
            with self.assertRaises(ValueError):
                validate_startup(trainer, augmentation)
            trainer.agent.scheduler = scheduler
            with self.assertRaises(ValueError):
                validate_startup(trainer, False)
            model.loss = nn.MSELoss()
            with self.assertRaises(ValueError):
                validate_startup(trainer, augmentation)
            trainer.env.close()


if __name__ == "__main__":
    unittest.main()
