"""Observation-only CSP/probability agent. No environment reference is accepted.

Apache-2.0 enumeration core: vendor/nabenabe_csp/UPSTREAM.md.
Input is a COPY of a 2-D visible board: -1 unknown, 0..8 revealed clue.
"""
from dataclasses import dataclass
from collections import Counter
import time

import numpy as np

from vendor.nabenabe_csp.probability import ProbabilityCalculator, MINE, NONE


class NoSolution(ValueError):
    pass


class SearchBudgetExceeded(RuntimeError):
    def __init__(self, reason, iterations):
        super().__init__(reason)
        self.reason = reason
        self.iterations = iterations


class BoundedCalculator(ProbabilityCalculator):
    """Keep upstream DFS; bound iterations and discard incomplete counts."""

    def __init__(self, *args, max_iterations, timeout_seconds, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.deadline = (time.perf_counter() + timeout_seconds
                         if timeout_seconds else None)
        self.iterations = 0

    def _add_count(self):
        # Defense in depth: never count a leaf violating any public clue.
        for i in self._opened_indices:
            actual = (self._n_flags_in_neighbors[i]
                      + np.count_nonzero(self._states[self._target_neighbors[i]] == MINE))
            if actual != self._cell_state[i]:
                raise ArithmeticError("Enumeration reached an inconsistent leaf")
        super()._add_count()

    def compute(self):
        if not self._states.size:
            raise NoSolution("Empty frontier is handled by the adapter")
        self._states[0], self._assumed[0] = MINE, True
        while np.any(self._assumed) or NONE in self._states:
            if self.iterations >= self.max_iterations:
                raise SearchBudgetExceeded("iteration_limit", self.iterations)
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                raise SearchBudgetExceeded("timeout", self.iterations)
            self.iterations += 1
            if not self._assume():
                self._update_target()
        if not self._total_count:
            raise NoSolution("No configuration satisfies local clues and total mines")
        self._target.proba = self._counts / self._total_count
        p_land = self._count4land / self._total_count if self._n_land_cells else 1.0
        return self._target, p_land


@dataclass
class Decision:
    action: int | None
    probabilities: np.ndarray
    reason: str
    exact: bool
    iterations: int = 0
    inferred_mines: tuple = ()


class CSPAgent:
    def __init__(self, rows=9, cols=9, num_mines=10,
                 max_iterations=20000, timeout_seconds=0.0):
        if rows <= 0 or cols <= 0 or not 0 <= num_mines < rows * cols:
            raise ValueError("Invalid board dimensions or mine count")
        if max_iterations < 1 or timeout_seconds < 0:
            raise ValueError("Positive iteration budget and nonnegative timeout required")
        self.rows, self.cols, self.num_mines = rows, cols, num_mines
        self.max_iterations, self.timeout_seconds = max_iterations, timeout_seconds
        self.neighbors = []
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            self.neighbors.append(np.array([
                rr * cols + cc
                for rr in range(max(0, r - 1), min(rows, r + 2))
                for cc in range(max(0, c - 1), min(cols, c + 2))
                if (rr, cc) != (r, c)
            ], dtype=np.int32))
        self.stats = Counter()
        # Public-observation diagnostics only; never store simulator truth.
        self.diagnostics = []

    def _record_no_solution(self, flat, unknown, mines, stage, message, iterations=0):
        constraints = []
        for i in np.flatnonzero(flat >= 0):
            adjacent = set(map(int, self.neighbors[i])) & unknown
            constraints.append({"clue_index": int(i), "clue": int(flat[i]),
                                "unknown_neighbors": sorted(adjacent),
                                "remaining_mines": int(flat[i]) - len(adjacent & mines)})
        self.diagnostics.append({
            "decision": self.stats["decisions"], "stage": stage, "message": str(message),
            "visible_board": flat.reshape(self.rows, self.cols).tolist(),
            "total_mines": self.num_mines, "remaining_mines": self.num_mines - len(mines),
            "inferred_mines": sorted(map(int, mines)), "constraints": constraints,
            "iterations": iterations})

    def _propagate(self, flat, unknown):
        """Cheap logically sound reductions before full CSP enumeration."""
        mines, safe = set(), set()
        while True:
            new_mines, new_safe = set(), set()
            for i in np.flatnonzero(flat >= 0):
                adjacent = set(self.neighbors[i]) & unknown
                variables = adjacent - mines - safe
                remaining = int(flat[i]) - len(adjacent & mines)
                if not 0 <= remaining <= len(variables):
                    raise NoSolution("Clue conflicts with inferred cells")
                if remaining == 0:
                    new_safe.update(variables)
                elif remaining == len(variables):
                    new_mines.update(variables)
            free = unknown - mines - safe
            remaining = self.num_mines - len(mines)
            if not 0 <= remaining <= len(free):
                raise NoSolution("Total mine count conflicts with clues")
            if remaining == 0:
                new_safe.update(free)
            elif remaining == len(free):
                new_mines.update(free)
            if (mines | new_mines) & (safe | new_safe):
                raise NoSolution("A cell is both safe and a mine")
            if not (new_mines - mines or new_safe - safe):
                return mines, safe
            mines.update(new_mines)
            safe.update(new_safe)

    @staticmethod
    def _choose(probabilities, candidates):
        # Equal within 1e-12: lowest row-major cell index, no RNG use.
        # A proven safe cell must win over any strictly positive guess,
        # even when that guess falls inside the numerical tie tolerance.
        certain = [i for i in candidates if probabilities[i] == 0.0]
        if certain:
            return min(certain)
        minimum = min(float(probabilities[i]) for i in candidates)
        return min(i for i in candidates if probabilities[i] <= minimum + 1e-12)

    def _fallback(self, flat, unknown, mines, reason, iterations=0):
        """Explicit heuristic, NOT a posterior or a proof of safety."""
        candidates = sorted(unknown - mines)
        if not candidates:
            self.stats["no_solution"] += 1
            self._record_no_solution(flat, unknown, mines, "fallback_candidates",
                                     "No unknown candidate outside inferred mines", iterations)
            raise NoSolution("No unknown candidate outside inferred mines; do not click an inferred mine")
        density = np.clip((self.num_mines - len(mines)) / len(candidates), 0, 1)
        probabilities = np.full(flat.size, np.nan)
        probabilities[candidates] = density
        probabilities[list(mines)] = 1.0
        estimates = {i: [] for i in candidates}
        for i in np.flatnonzero(flat >= 0):
            adjacent = set(self.neighbors[i]) & unknown
            free = adjacent - mines
            remaining = int(flat[i]) - len(adjacent & mines)
            if free and 0 <= remaining <= len(free):
                for j in free:
                    if j in estimates:
                        estimates[j].append(remaining / len(free))
        for i, values in estimates.items():
            if values:
                probabilities[i] = sum(values) / len(values)
        self.stats["fallbacks"] += 1
        self.stats["heuristic_fallbacks"] += 1
        self.stats[reason] += 1
        self.stats["enumeration_iterations"] += iterations
        return Decision(self._choose(probabilities, candidates), probabilities,
                        reason, False, iterations, tuple(sorted(mines)))

    def decide(self, visible_board):
        board = np.array(visible_board, dtype=np.int32, copy=True)
        if board.shape != (self.rows, self.cols):
            raise ValueError("Expected a 2-D visible board of the declared dimensions")
        if np.any((board < -1) | (board > 8)):
            raise ValueError("Agent accepts live observations only: -1 or 0..8")
        flat = board.reshape(-1)
        unknown = set(map(int, np.flatnonzero(flat == -1)))
        self.stats["decisions"] += 1
        probabilities = np.full(flat.size, np.nan)
        if not unknown:
            return Decision(None, probabilities, "no_unknown", True)
        if len(unknown) == flat.size:
            # Environment handles first-click safety; opening is always index 0.
            # Conditional on the fixed first cell being safe after regeneration.
            probabilities[:] = self.num_mines / max(1, flat.size - 1)
            probabilities[0] = 0.0
            self.stats["first_click"] += 1
            return Decision(0, probabilities, "first_click", True)
        try:
            mines, safe = self._propagate(flat, unknown)
        except NoSolution as exc:
            self._record_no_solution(flat, unknown, set(), "propagation", exc)
            return self._fallback(flat, unknown, set(), "no_solution")
        probabilities[list(mines)] = 1.0
        if safe:
            probabilities[list(safe)] = 0.0
            self.stats["deduced_safe"] += 1
            return Decision(min(safe), probabilities, "deduced_safe", True,
                            inferred_mines=tuple(sorted(mines)))
        flags = np.zeros(flat.size, dtype=np.bool_)
        flags[list(mines)] = True
        calc = BoundedCalculator(flat, flags, self.neighbors, self.num_mines,
                                 max_iterations=self.max_iterations,
                                 timeout_seconds=self.timeout_seconds)
        candidates = sorted(unknown - mines)
        if not calc._states.size:
            # No frontier: exact uniform probability given the remaining budget.
            probabilities[candidates] = (self.num_mines - len(mines)) / len(candidates)
            self.stats["no_constraints"] += 1
            self.stats["fallbacks"] += 1
            return Decision(self._choose(probabilities, candidates), probabilities,
                            "no_constraints", True, inferred_mines=tuple(sorted(mines)))
        try:
            target, p_land = calc.compute()
        except SearchBudgetExceeded as exc:
            return self._fallback(flat, unknown, mines, exc.reason, exc.iterations)
        except NoSolution as exc:
            self._record_no_solution(flat, unknown, mines, "enumeration", exc, calc.iterations)
            return self._fallback(flat, unknown, mines, "no_solution", calc.iterations)
        probabilities[target.index] = target.proba
        probabilities[calc._is_land & ~flags] = p_land
        p = probabilities[list(unknown)]
        if not np.isfinite(p).all() or np.any((p < -1e-12) | (p > 1 + 1e-12)):
            raise ArithmeticError("Invalid enumerated probabilities")
        if not np.isclose(p.sum(), self.num_mines, atol=1e-7, rtol=0):
            raise ArithmeticError("Posterior mine mass differs from public total")
        action = self._choose(probabilities, candidates)
        self.stats["enumerations"] += 1
        self.stats["enumeration_iterations"] += calc.iterations
        inferred = tuple(sorted(i for i in unknown if probabilities[i] == 1.0))
        return Decision(action, probabilities, "enumerated", True,
                        calc.iterations, inferred)
