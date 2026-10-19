"""Synthetic checkpoints test compatibility; these are NOT research results."""
import tempfile
from pathlib import Path
import unittest

import numpy as np


class CNNAdapterTests(unittest.TestCase):
    def test_retained_and_removed_dropout_checkpoint_layouts(self):
        import torch
        from lib import models
        from baseline_evaluation import CNNPolicy
        torch.set_num_threads(2)
        torch.manual_seed(917)
        visible = np.array([[0, 1, -1, -1], [1, 2, -1, -1], [-1, -1, -1, -1]])
        for layout in ("retained", "removed"):
            model = models.Model((10, 3, 4), 12)
            if layout == "removed":
                model.fc = torch.nn.Sequential(*[
                    layer for layer in model.fc if not isinstance(layer, torch.nn.Dropout)
                ])
            model.eval()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "synthetic.dat"
                torch.save(model.state_dict(), path)
                policy = CNNPolicy(3, 4, path, "cpu", layout)
                first = policy.decide(visible)
                second = policy.decide(visible)
                self.assertEqual(first.action, second.action)
                self.assertEqual(visible.reshape(-1)[first.action], -1)
                state = np.zeros((10, 3, 4), dtype=np.float32)
                for number in range(1, 9):
                    state[number - 1] = visible == number
                state[8] = visible == 0
                state[9] = visible == -1
                with torch.no_grad():
                    expected = model(torch.from_numpy(state).unsqueeze(0)).numpy().reshape(-1)
                expected[visible.reshape(-1) != -1] = np.inf
                np.testing.assert_array_equal(first.probabilities, expected)


if __name__ == "__main__":
    unittest.main()
