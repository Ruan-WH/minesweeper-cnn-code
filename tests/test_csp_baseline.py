"""Brute-force oracle and original-environment integration checks."""
import itertools
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from lib.csp_agent import CSPAgent, BoundedCalculator, NoSolution, SearchBudgetExceeded


def make_board(agent, mines):
    truth = np.zeros(agent.rows * agent.cols, dtype=np.int32)
    for i, neighbors in enumerate(agent.neighbors):
        truth[i] = -2 if i in mines else len(set(neighbors) & mines)
    return truth.reshape(agent.rows, agent.cols)


def oracle(agent, visible):
    flat = visible.reshape(-1)
    unknown = list(map(int, np.flatnonzero(flat == -1)))
    valid = []
    for cells in itertools.combinations(unknown, agent.num_mines):
        mines = set(cells)
        if all(len(set(agent.neighbors[i]) & mines) == flat[i]
               for i in np.flatnonzero(flat >= 0)):
            valid.append(mines)
    if not valid:
        return None
    return {i: sum(i in mines for mines in valid) / len(valid) for i in unknown}


class CSPTests(unittest.TestCase):
    def test_combination_weighted_five_variable_example(self):
        # A,B,C,D,E and two public clues: A+B=1, B+C=1.
        board = np.array([-1, -1, -1, -1, -1, 1, 1])
        neighbors = [np.array(v, dtype=np.int32) for v in
                     ([5], [5, 6], [6], [], [], [0, 1], [1, 2])]
        calc = BoundedCalculator(board, np.zeros(7, dtype=bool), neighbors, 2,
                                 max_iterations=10000, timeout_seconds=0)
        target, p_land = calc.compute()
        np.testing.assert_array_equal(target.index, [0, 1, 2])
        np.testing.assert_allclose(target.proba, [1/3, 2/3, 1/3], atol=1e-12)
        self.assertAlmostEqual(p_land, 1/3)
        self.assertAlmostEqual(calc._total_count, 3)
        self.assertAlmostEqual(target.proba.sum() + 2*p_land, 2)

    def test_strict_rules_and_global_mine_deduction(self):
        agent = CSPAgent(1, 4, 1)
        flat = np.array([1, -1, -1, -1])
        mines, safe = agent._propagate(flat, {1, 2, 3})
        self.assertEqual(mines, {1})
        self.assertEqual(safe, {2, 3})
        decision = agent.decide(flat.reshape(1, 4))
        self.assertEqual(decision.action, 2)
        self.assertEqual(decision.inferred_mines, (1,))
        # A zero clue establishes safety without enumeration.
        zero = CSPAgent(1, 4, 1).decide(np.array([[0, -1, -1, -1]]))
        self.assertEqual(zero.action, 1)
        self.assertEqual(zero.reason, 'deduced_safe')

    def test_incomplete_enumeration_is_never_used_as_probability(self):
        def partial_then_stop(calc):
            calc._counts[:] = 999
            calc._total_count = 999
            calc._target.proba[:] = 0
            raise SearchBudgetExceeded('iteration_limit', 17)
        visible = np.full((9, 9), -1)
        visible[0, 0] = 1
        agent = CSPAgent(num_mines=10)
        with patch.object(BoundedCalculator, 'compute', partial_then_stop):
            decision = agent.decide(visible)
        self.assertFalse(decision.exact)
        self.assertEqual(decision.reason, 'iteration_limit')
        self.assertAlmostEqual(decision.probabilities[1], 1/3)
        self.assertAlmostEqual(decision.probabilities[80], 10/80)
        p = decision.probabilities[visible.reshape(-1) == -1]
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        self.assertEqual(agent.stats['heuristic_fallbacks'], 1)

    def test_no_solution_diagnostics_and_empty_candidate_guard(self):
        agent = CSPAgent(2, 2, 1)
        visible = np.array([[3, -1], [0, -1]])
        decision = agent.decide(visible)
        self.assertFalse(decision.exact)
        self.assertEqual(decision.probabilities[decision.action], 0.0)
        self.assertEqual(len(agent.diagnostics), 1)
        self.assertEqual(agent.diagnostics[0]['visible_board'], visible.tolist())
        self.assertIn('constraints', agent.diagnostics[0])
        with self.assertRaises(NoSolution):
            agent._fallback(visible.reshape(-1), {1, 3}, {1, 3}, 'no_solution')

    def test_no_frontier_is_exact_uniform_not_heuristic(self):
        agent = CSPAgent(1, 5, 2)
        decision = agent.decide(np.array([[0, 1, -1, -1, -1]]))
        self.assertEqual(decision.reason, 'no_constraints')
        self.assertTrue(decision.exact)
        self.assertEqual(decision.inferred_mines, (2,))
        self.assertEqual(decision.action, 3)
        np.testing.assert_allclose(decision.probabilities[2:], [1, 0.5, 0.5])
        self.assertEqual(agent.stats['no_constraints'], 1)
        self.assertEqual(agent.stats['heuristic_fallbacks'], 0)

    def test_harness_writes_public_no_solution_diagnostics(self):
        from baseline_evaluation import main
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(CSPAgent, '_propagate', side_effect=NoSolution('injected audit test')):
                result = main(['--mines', '18', '--episodes', '1', '--seed', '20261013',
                               '--output', directory])
            diagnostic_path = Path(directory) / 'no_solution.jsonl'
            records = [json.loads(line) for line in diagnostic_path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), result[0]['counts']['no_solution'])
            self.assertGreater(len(records), 0)
            self.assertEqual(result[0]['errors'], 0)
            self.assertEqual(result[0]['games_with_no_solution'], 1)
            self.assertEqual(result[0]['games_with_solver_issue'], 1)
            self.assertEqual(result[0]['games_with_heuristic_fallback'], 1)
            self.assertEqual(result[0]['no_solution_decision_rate'], 1.0)
            self.assertEqual(result[0]['no_constraints_decision_rate'], 0.0)
            for record in records:
                self.assertEqual(record['seed'], 20261013)
                self.assertEqual(record['total_mines'], 18)
                self.assertTrue(all(-1 <= cell <= 8 for row in record['visible_board'] for cell in row))

    def test_visible_state_matches_cnn_encoding_without_truth_access(self):
        from baseline_evaluation import play_episode
        from minesweeper_pygame import MinesweeperEnv
        from lib.wrappers import CustomObservationWrapper
        wrapper = CustomObservationWrapper(MinesweeperEnv(9, 9, 18, 'rgb_array'), 'one_hot')
        agent = CSPAgent(num_mines=18)
        original_decide = agent.decide
        seen = []
        def checked_decide(visible):
            self.assertTrue(np.all((visible >= -1) & (visible <= 8)))
            self.assertGreaterEqual(visible[0, 0], 0)
            encoded = wrapper.observation(visible)
            np.testing.assert_array_equal(encoded[-1] == 1, visible == -1)
            decoded = np.full((9, 9), -1)
            for digit in range(1, 9):
                decoded[encoded[digit-1] == 1] = digit
            decoded[encoded[8] == 1] = 0
            np.testing.assert_array_equal(decoded, visible)
            seen.append(visible.copy())
            return original_decide(visible)
        agent.decide = checked_decide
        class BoardDeniedEnv:
            def __init__(self, base):
                self.base = base
                self.actions = []
            @property
            def board(self):
                raise AssertionError('Hidden board accessed outside audit helper')
            @property
            def button(self):
                return self.base.button
            @button.setter
            def button(self, value):
                self.base.button = value
            def __getattr__(self, name):
                return getattr(self.base, name)
            def step(self, action):
                self.actions.append(action)
                self.assert_unknown(action)
                return self.base.step(action)
            def assert_unknown(self, action):
                assert self.base.player_board.reshape(-1)[action] == -1
        env = BoardDeniedEnv(wrapper.unwrapped)
        with patch('baseline_evaluation.audit_board_hash', return_value='audit_disabled_for_isolation_test'):
            row = play_episode(env, agent, 20261013, 0, 18)
        self.assertEqual(row['error'], '')
        self.assertEqual(env.actions[0], 0)
        self.assertEqual(env.actions.count(0), 1)
        self.assertEqual(len(env.actions), len(set(env.actions)))
        self.assertTrue(seen)
        wrapper.close()

    def test_budget_fallback_continues_original_game(self):
        from baseline_evaluation import play_episode
        from minesweeper_pygame import MinesweeperEnv
        env = MinesweeperEnv(9, 9, 22, 'rgb_array')
        row = play_episode(env, CSPAgent(num_mines=22, max_iterations=1), 20261013, 0, 22)
        self.assertEqual(row['error'], '')
        self.assertIn(row['status'], ('win', 'lose'))
        self.assertGreater(row['iteration_limit'], 0)
        self.assertGreater(row['steps'], 2)
        self.assertEqual(row['status'], env.info['status'])
        env.close()

    def test_probabilities_against_exhaustive_oracle(self):
        rng = np.random.RandomState(931)
        enumerations = 0
        nontrivial_safe = 0
        for _ in range(500):
            agent = CSPAgent(3, 4, 3, max_iterations=100000)
            mines = set(map(int, rng.choice(12, 3, replace=False)))
            truth = make_board(agent, mines)
            visible = np.full((3, 4), -1, dtype=np.int32)
            safe = sorted(set(range(12)) - mines)
            revealed = rng.choice(safe, rng.randint(1, len(safe)), replace=False)
            visible.reshape(-1)[revealed] = truth.reshape(-1)[revealed]
            expected = oracle(agent, visible)
            self.assertIsNotNone(expected)
            decision = agent.decide(visible)
            self.assertIn(decision.action, expected)
            self.assertTrue(decision.exact)
            for i, p in expected.items():
                if np.isfinite(decision.probabilities[i]):
                    self.assertAlmostEqual(decision.probabilities[i], p, places=10)
            if decision.reason == "enumerated":
                enumerations += 1
                self.assertAlmostEqual(expected[decision.action], min(expected.values()), places=10)
                if expected[decision.action] == 0:
                    nontrivial_safe += 1
        self.assertGreater(enumerations, 10)
        self.assertGreater(nontrivial_safe, 0)  # proves inference beyond single-point rules

    def test_budget_and_timeout_discard_partial_counts(self):
        visible = np.full((9, 9), -1, dtype=np.int32)
        visible[0, 0] = 1
        limited = CSPAgent(num_mines=10, max_iterations=1).decide(visible)
        self.assertEqual(limited.reason, "iteration_limit")
        self.assertFalse(limited.exact)
        timed = CSPAgent(num_mines=10, timeout_seconds=1e-15).decide(visible)
        self.assertEqual(timed.reason, "timeout")
        self.assertEqual(visible.reshape(-1)[timed.action], -1)

    def test_upstream_propagation_regression(self):
        agent = CSPAgent(3, 4, 3)
        visible = np.array([[-1, -1, -1, -1], [-1, -1, 3, 2], [1, -1, -1, 1]])
        decision = agent.decide(visible)
        expected = oracle(agent, visible)
        self.assertEqual(decision.reason, "enumerated")
        for i, p in expected.items():
            self.assertAlmostEqual(decision.probabilities[i], p, places=12)
        self.assertEqual(decision.probabilities[4], 0)

    def test_globally_unsatisfiable_enumeration(self):
        visible = np.array([[-1, 1, -1], [1, -1, 1], [-1, 1, -1]])
        agent = CSPAgent(3, 3, 3)
        self.assertIsNone(oracle(agent, visible))
        decision = agent.decide(visible)
        self.assertEqual(decision.reason, "no_solution")
        self.assertGreater(decision.iterations, 0)
        self.assertFalse(decision.exact)

    def test_inconsistent_clues_and_no_unknown(self):
        agent = CSPAgent(2, 2, 1)
        decision = agent.decide(np.array([[3, -1], [0, -1]]))
        self.assertEqual(decision.reason, "no_solution")
        self.assertFalse(decision.exact)
        self.assertIn(decision.action, (1, 3))
        self.assertIsNone(agent.decide(np.zeros((2, 2), dtype=int)).action)
        with self.assertRaises(ValueError):
            agent.decide(np.array([[-2, -1], [-1, -1]]))

    def test_interface_is_observation_only_and_deterministic(self):
        visible = np.full((9, 9), -1, dtype=np.int32)
        self.assertEqual(CSPAgent().decide(visible).action, 0)
        visible[0, 0] = 1
        before = visible.copy()
        a, b = CSPAgent().decide(visible), CSPAgent().decide(visible)
        self.assertEqual(a.action, b.action)
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        np.testing.assert_array_equal(visible, before)
        self.assertFalse(hasattr(CSPAgent(), "env"))

    def test_proven_safe_priority_and_first_click_probability(self):
        self.assertEqual(CSPAgent._choose(np.array([1e-15, 0.0]), [0, 1]), 1)
        decision = CSPAgent().decide(np.full((9, 9), -1))
        self.assertEqual(decision.probabilities[0], 0.0)
        self.assertAlmostEqual(decision.probabilities.sum(), 10.0)

    def test_original_first_click_and_terminal_states(self):
        from minesweeper_pygame import MinesweeperEnv
        env = MinesweeperEnv(9, 9, 10, "rgb_array")
        positive_first_clues = 0
        for seed in range(30):
            np.random.seed(seed)
            env.reset()
            state, _, done, info = env.step(0)
            self.assertNotEqual(state[0, 0], -2)
            self.assertNotEqual(info["status"], "lose")
            positive_first_clues += int(state[0, 0] > 0)
        self.assertGreater(positive_first_clues, 0)
        # Ground truth access in TEST FIXTURES only: force a loss and finish a win.
        np.random.seed(42)
        env.reset()
        env.step(0)
        mine = int(np.flatnonzero(env.board.reshape(-1) == -2)[0])
        _, _, done, info = env.step(mine)
        self.assertTrue(done)
        self.assertEqual(info["status"], "lose")
        self.assertEqual(env.stats["n_lose"], 1)
        env.reset()
        env.step(0)
        while not env.done:
            safe = np.flatnonzero((env.player_board.reshape(-1) == -1)
                                  & (env.board.reshape(-1) != -2))
            _, _, done, info = env.step(int(safe[0]))
        self.assertEqual(info["status"], "win")
        self.assertEqual(env.stats["n_win"], 1)
        env.close()

    def test_episode_seed_reproducibility_and_audit_isolation(self):
        from baseline_evaluation import play_episode
        from minesweeper_pygame import MinesweeperEnv
        class OtherPolicy:
            # Changes actions but has the same observed first click.
            stats = {}
            def decide(self, visible):
                from lib.csp_agent import Decision
                unknown = np.flatnonzero(visible.reshape(-1) == -1)
                return Decision(int(unknown[-1]), np.zeros(visible.size), "test", False)
        for seed in range(10):
            env = MinesweeperEnv(9, 9, 18, "rgb_array")
            a = play_episode(env, CSPAgent(num_mines=18), seed, seed, 18)
            b = play_episode(env, CSPAgent(num_mines=18), seed, seed, 18)
            for key in a:
                if key not in ("seconds", "decision_seconds", "mean_decision_seconds"):
                    self.assertEqual(a[key], b[key])
            c = play_episode(env, OtherPolicy(), seed, seed, 18)
            self.assertEqual(a["initial_board_sha256"], c["initial_board_sha256"])
            self.assertEqual(a["first_click_board_sha256"], c["first_click_board_sha256"])
            self.assertEqual(a["error"], "")
            env.close()


if __name__ == "__main__":
    unittest.main()
