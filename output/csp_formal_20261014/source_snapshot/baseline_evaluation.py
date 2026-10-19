"""Shared, paired-board test harness for CSP and fixed CNN checkpoints.

Hidden-board hashing is restricted to the audit helper. Agents receive only
copies of visible observations. Neither the original environment nor trainers
are modified.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from lib.csp_agent import CSPAgent

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "3a499ddee44bf7562808c8ab787bd173fb68fa2e"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_board_hash(env):
    """Harness ONLY. Hash is written to audit logs, never passed to agents."""
    return hashlib.sha256(np.asarray(env.board, dtype="<i4").tobytes()).hexdigest()


def wilson_interval(wins, episodes):
    if episodes <= 0:
        raise ValueError("At least one episode is required")
    z = 1.959963984540054
    p = wins / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes ** 2)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


class CNNPolicy:
    """Same visible encoding and greedy decision as original offline CNN test."""
    def __init__(self, rows, cols, model_path, device_name, dropout_layout):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from lib import models
        self.torch = torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.device = torch.device(device_name)
        self.net = models.Model((10, rows, cols), rows * cols).to(self.device)
        if dropout_layout == "removed":
            # Supports checkpoints trained with the user's commented-out layers.
            self.net.fc = torch.nn.Sequential(*[
                layer for layer in self.net.fc if not isinstance(layer, torch.nn.Dropout)
            ])
        self.net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.net.eval()
        self.stats = Counter()

    def decide(self, visible_board):
        from lib.csp_agent import Decision
        torch = self.torch
        state = np.zeros((10, *visible_board.shape), dtype=np.float32)
        for number in range(1, 9):
            state[number - 1] = visible_board == number
        state[8] = visible_board == 0
        state[9] = visible_board == -1
        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            scores = self.net(tensor).cpu().numpy().reshape(-1).copy()
        scores[visible_board.reshape(-1) != -1] = np.inf
        action = int(np.argmin(scores))
        if not np.isfinite(scores[action]):
            raise ValueError("No finite CNN score for a legal unknown cell")
        return Decision(action, scores, "cnn", False)


def play_episode(env, policy, seed, episode, num_mines):
    # Reset each episode's environment RNG. Policy consumes no global numpy RNG.
    np.random.seed(seed)
    visible = np.array(env.reset(), copy=True)
    initial_hash = audit_board_hash(env)
    first_hash = ""
    started = time.perf_counter()
    before = policy.stats.copy()
    done, info, steps = False, {}, 0
    reasons = Counter()
    decision_seconds = 0.0
    decision_count = 0
    max_search_iterations = 0
    exact_decisions = 0
    diagnostic_start = len(getattr(policy, "diagnostics", []))
    action_digest = hashlib.sha256()
    error = ""
    try:
        for step in range(env.num_rows * env.num_cols):
            # Identical to the existing CNN offline protocol.
            if step == 0:
                action = 0
            else:
                decision_started = time.perf_counter()
                decision = policy.decide(visible.copy())
                decision_seconds += time.perf_counter() - decision_started
                decision_count += 1
                max_search_iterations = max(max_search_iterations, decision.iterations)
                exact_decisions += int(decision.exact)
                action = decision.action
                reasons[decision.reason] += 1
                if action in decision.inferred_mines:
                    raise ValueError("Policy selected an internally inferred mine")
            if action is None or not 0 <= action < visible.size or visible.reshape(-1)[action] != -1:
                raise ValueError(f"Illegal action: {action}")
            env.button = "left"
            next_state, reward, done, info = env.step(int(action))
            visible = np.array(next_state, copy=True)
            steps += 1
            action_digest.update(int(action).to_bytes(4, "little"))
            if step == 0:
                first_hash = audit_board_hash(env)
                if info.get("status") == "lose" or visible[0, 0] == -2:
                    raise AssertionError("Original first-click safety was violated")
            if done:
                if info.get("status") not in ("win", "lose"):
                    raise ValueError("Unexpected terminal status")
                break
        if not done:
            raise RuntimeError("Episode exceeded board-area step bound")
    except Exception as exc:
        # Never silently drop a failed game: it remains in the denominator.
        error = f"{type(exc).__name__}: {exc}"
    counts = {key: value - before[key] for key, value in policy.stats.items()
              if value - before[key]}
    return {
        "mines": num_mines, "episode": episode, "seed": seed,
        "status": "error" if error else info["status"],
        "steps": steps, "seconds": time.perf_counter() - started,
        "decision_count": decision_count, "decision_seconds": decision_seconds,
        "mean_decision_seconds": decision_seconds / decision_count if decision_count else 0.0,
        "max_search_iterations": max_search_iterations, "exact_decisions": exact_decisions,
        "initial_board_sha256": initial_hash, "first_click_board_sha256": first_hash,
        "actions_sha256": action_digest.hexdigest(), "error": error,
        "fallbacks": counts.get("fallbacks", 0),
        "heuristic_fallbacks": counts.get("heuristic_fallbacks", 0),
        "iteration_limit": counts.get("iteration_limit", 0),
        "timeouts": counts.get("timeout", 0),
        "no_solution": counts.get("no_solution", 0),
        "no_constraints": counts.get("no_constraints", 0),
        "enumerations": counts.get("enumerations", 0),
        "enumeration_iterations": counts.get("enumeration_iterations", 0),
        "decision_reasons": json.dumps(dict(reasons), sort_keys=True),
        "no_solution_diagnostics": json.dumps(getattr(policy, "diagnostics", [])[diagnostic_start:]),
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--agent", choices=("csp", "cnn"), default="csp")
    result.add_argument("--mines", nargs="+", type=int, default=[10, 14, 18, 22])
    result.add_argument("--episodes", type=int, default=100)
    result.add_argument("--seed", type=int, default=20261013)
    result.add_argument("--max-iterations", type=int, default=20000)
    result.add_argument("--timeout-seconds", type=float, default=0.0,
                        help="0 disables wall-clock cutoff; iteration budget is always active")
    result.add_argument("--output", type=Path, default=None)
    result.add_argument("--model-path", type=Path)
    result.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    result.add_argument("--cnn-dropout", choices=("retained", "removed"), default="retained",
                        help="Checkpoint architecture: retain Dropout slots or remove them")
    result.add_argument("--verify-boards", type=Path,
                        help="Reference episodes.csv; abort on any paired-board mismatch")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.episodes < 1 or not 0 <= args.seed < 2 ** 32:
        raise ValueError("Positive episodes and uint32 seed required")
    if len(set(args.mines)) != len(args.mines) or any(m not in (10, 14, 18, 22) for m in args.mines):
        raise ValueError("Choose unique mine counts from 10, 14, 18, 22")
    if args.agent == "cnn" and (args.model_path is None or len(args.mines) != 1):
        raise ValueError("CNN requires one mine count and its matching --model-path")
    # Validate budgets even before the first episode.
    CSPAgent(max_iterations=args.max_iterations, timeout_seconds=args.timeout_seconds)
    from minesweeper_pygame import MinesweeperEnv

    output = args.output or Path("output/csp_baseline") / f"{args.agent}-{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "episodes.csv"
    if csv_path.exists() or (output / "summary.json").exists():
        raise FileExistsError("Use a new output directory to preserve existing results")
    reference = None
    if args.verify_boards:
        with args.verify_boards.open(encoding="utf-8", newline="") as stream:
            reference = {(int(row["mines"]), int(row["episode"])): row
                         for row in csv.DictReader(stream)}
    metadata = {
        "agent": args.agent, "rows": 9, "cols": 9, "mines": args.mines,
        "episodes_per_mine_count": args.episodes, "seed": args.seed,
        "episode_seed_rule": "(seed + zero_based_episode_index) modulo 2**32",
        "first_click": 0, "first_click_rule": "original environment rejection regeneration; safe, not necessarily zero",
        "max_iterations_per_decision": args.max_iterations,
        "timeout_seconds_per_decision": args.timeout_seconds,
        "tie_break": "proven safe first; otherwise lowest row-major index within absolute probability tolerance 1e-12",
        "confidence_interval": "two-sided Wilson 95%",
        "upstream_commit": UPSTREAM_COMMIT,
        "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "dependencies": {},
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "lib/csp_agent.py", "vendor/nabenabe_csp/probability.py",
            "vendor/nabenabe_csp/constants.py", "baseline_evaluation.py",
            "minesweeper_pygame.py", "constants.py", "lib/models.py")},
        "model_path": str(args.model_path) if args.model_path else None,
        "model_sha256": sha256_file(args.model_path) if args.model_path else None,
        "cnn_dropout_layout": args.cnn_dropout if args.agent == "cnn" else None,
        "device": args.device if args.agent == "cnn" else "cpu",
        "cnn_deterministic_algorithms": args.agent == "cnn",
        "paired_reference": str(args.verify_boards) if args.verify_boards else None,
    }
    for package in ("numpy", "gym", "pygame", "torch"):
        try:
            metadata["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["dependencies"][package] = None
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summaries, fields = [], None
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        for mines in args.mines:
            env = MinesweeperEnv(9, 9, mines, "rgb_array")
            policy = (CSPAgent(9, 9, mines, args.max_iterations, args.timeout_seconds)
                      if args.agent == "csp" else CNNPolicy(9, 9, args.model_path, args.device, args.cnn_dropout))
            started = time.perf_counter()
            rows = []
            try:
                for episode in range(args.episodes):
                    seed = (args.seed + episode) % (2 ** 32)
                    row = play_episode(env, policy, seed, episode, mines)
                    diagnostics = json.loads(row["no_solution_diagnostics"])
                    if diagnostics:
                        with (output / "no_solution.jsonl").open("a", encoding="utf-8") as diagnostic_stream:
                            for diagnostic in diagnostics:
                                diagnostic_stream.write(json.dumps({"mines": mines, "episode": episode,
                                    "seed": seed, **diagnostic}) + "\n")
                    if reference is not None:
                        old = reference.get((mines, episode))
                        if old is None or any(str(row[key]) != old[key] for key in (
                                "seed", "initial_board_sha256", "first_click_board_sha256")):
                            raise AssertionError(f"Paired board mismatch: mines={mines}, episode={episode}")
                    if fields is None:
                        fields = csv.DictWriter(stream, fieldnames=list(row))
                        fields.writeheader()
                    fields.writerow(row)
                    stream.flush()
                    rows.append(row)
                    if (episode + 1) % 100 == 0:
                        print(f"{args.agent}: mines={mines} {episode + 1}/{args.episodes}, "
                              f"wins={sum(r['status'] == 'win' for r in rows)}", flush=True)
            finally:
                env.close()
            wins = sum(row["status"] == "win" for row in rows)
            decisions = sum(row["decision_count"] for row in rows)
            heuristic = sum(row["heuristic_fallbacks"] for row in rows)
            enumerations = sum(row["enumerations"] for row in rows)
            limited = sum(row["iteration_limit"] for row in rows)
            summary = {"mines": mines, "wins": wins, "episodes": len(rows),
                       "win_rate": wins / len(rows), "ci95": wilson_interval(wins, len(rows)),
                       "seconds": time.perf_counter() - started,
                       "losses": sum(row["status"] == "lose" for row in rows),
                       "errors": sum(row["status"] == "error" for row in rows),
                       "games_with_fallback": sum(row["fallbacks"] > 0 for row in rows),
                       "games_with_budget_exhaustion": sum(row["iteration_limit"] > 0 for row in rows),
                       "games_with_timeout": sum(row["timeouts"] > 0 for row in rows),
                       "games_with_no_solution": sum(row["no_solution"] > 0 for row in rows),
                       "games_with_no_constraints": sum(row["no_constraints"] > 0 for row in rows),
                       "games_with_heuristic_fallback": sum(row["heuristic_fallbacks"] > 0 for row in rows),
                       "games_with_solver_issue": sum(
                           any(row[key] > 0 for key in ("iteration_limit", "timeouts", "no_solution"))
                           for row in rows),
                       "mean_game_seconds": sum(row["seconds"] for row in rows) / len(rows),
                       "decision_count": decisions,
                       "mean_decision_seconds": sum(row["decision_seconds"] for row in rows) / decisions if decisions else 0.0,
                       "max_search_iterations": max(row["max_search_iterations"] for row in rows),
                       "iteration_limit_decision_rate": limited / decisions if decisions else 0.0,
                       "enumeration_success_decision_rate": enumerations / decisions if decisions else 0.0,
                       "heuristic_fallback_decision_rate": heuristic / decisions if decisions else 0.0,
                       "timeout_decision_rate": sum(row["timeouts"] for row in rows) / decisions if decisions else 0.0,
                       "no_solution_decision_rate": sum(row["no_solution"] for row in rows) / decisions if decisions else 0.0,
                       "no_constraints_decision_rate": sum(row["no_constraints"] for row in rows) / decisions if decisions else 0.0,
                       "counts": dict(policy.stats)}
            summaries.append(summary)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    if any(summary["errors"] for summary in summaries):
        raise SystemExit("Some games failed with errors; inspect episodes.csv before reporting results")
    return summaries


if __name__ == "__main__":
    main()
