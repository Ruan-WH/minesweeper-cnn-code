"""Mixed adaptation using the unchanged training Agent and current model.

The reference module is loaded through its AST to avoid its import-time model/log
directory creation and main-loop side effects. No loss, Dropout, augmentation or
scheduler is silently enabled. Those settings remain in the user's original files.
"""
import argparse
import ast
from collections import Counter, deque
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import types

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from constants import BrickState
from lib import models
from lib.wrappers import CustomObservationWrapper
from minesweeper_pygame import MinesweeperEnv

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path, output):
    """Reuse original class/functions verbatim, without executing its top level."""
    source = Path(path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in (
                "Agent", "augment_data", "_convert_board"):
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LEARNING_RATE"
                for target in node.targets):
            selected.append(node)
    module = types.ModuleType("mixed_training_reference")
    module.__dict__.update(os=os, copy=copy, torch=torch, optim=optim,
                           DataLoader=DataLoader, TensorDataset=TensorDataset,
                           time=time, datetime=datetime, np=np, models=models,
                           BrickState=BrickState, lr_scheduler=lr_scheduler,
                           log_dir=str(output), model_dir=str(output))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    for name in ("Agent", "augment_data", "_convert_board", "LEARNING_RATE"):
        if not hasattr(module, name):
            raise ValueError(f"Reference script lacks {name}")
    agent_node = next(n for n in selected if isinstance(n, ast.ClassDef) and n.name == "Agent")
    init = next(n for n in agent_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],
                        [ast.literal_eval(d) for d in init.args.defaults]))
    if defaults.get("num_samples") != 1024 or defaults.get("epochs") != 10:
        raise ValueError("Reference must use the requested 1024 samples / 10 epochs")
    expressions = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    if target.attr in ("optimizer", "scheduler"):
                        expressions[target.attr] = compile(ast.Expression(node.value), str(path), "eval")
    if "optimizer" not in expressions:
        raise ValueError("Reference optimizer assignment not found")
    update = next(n for n in agent_node.body if isinstance(n, ast.FunctionDef) and n.name == "update_model")
    batches = [ast.literal_eval(kw.value) for n in ast.walk(update) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == "DataLoader"
               for kw in n.keywords if kw.arg == "batch_size"]
    if batches != [32]:
        raise ValueError("Reference must use one DataLoader with batch_size=32")
    augmentation = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "augment_data" for n in ast.walk(update))
    return module, expressions, source, augmentation


def prepare_mixed_reference(reference, augmentation):
    """Mixed-only startup configuration; leave the reference file unchanged."""
    original_augment = reference.augment_data
    reference.augmentation_calls = 0
    def tracked_augment(*args, **kwargs):
        reference.augmentation_calls += 1
        return original_augment(*args, **kwargs)
    reference.augment_data = tracked_augment
    if not augmentation:
        original_update = reference.Agent.update_model
        def augmented_update(agent):
            features, labels = agent.features, agent.labels
            agent.features, agent.labels = reference.augment_data(
                features, labels, agent.env.num_rows, agent.env.num_cols)
            try:
                return original_update(agent)
            finally:
                agent.features, agent.labels = features, labels
        reference.Agent.update_model = augmented_update
    return True


def configure_mixed_loss(model):
    # Instance-only override: Fixed/SCL Model defaults remain untouched.
    model.loss = torch.nn.BCEWithLogitsLoss(weight=None, pos_weight=None, reduction="mean")


def validate_startup(trainer, augmentation):
    loss = trainer.model.loss
    if not isinstance(loss, torch.nn.BCEWithLogitsLoss) or loss.weight is not None or loss.pos_weight is not None or loss.reduction != "mean":
        raise ValueError("Mixed requires unweighted mean BCEWithLogitsLoss")
    dropout = [layer.p for layer in trainer.model.modules() if isinstance(layer, torch.nn.Dropout)]
    if dropout != [0.3, 0.3]:
        raise ValueError(f"SCL Dropout must be [0.3, 0.3], got {dropout}")
    if not augmentation:
        raise ValueError("Mixed requires paired feature/label augmentation")
    scheduler = getattr(trainer.agent, "scheduler", None)
    if not isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
        raise ValueError("Mixed requires active reference CosineAnnealingLR and scheduler.step()")
    return {"loss": type(loss).__name__, "loss_weight": None, "pos_weight": None,
            "reduction": loss.reduction, "dropout": dropout, "augmentation": True,
            "optimizer": type(trainer.agent.optimizer).__name__,
            "optimizer_defaults": trainer.agent.optimizer.defaults,
            "initial_lr": scheduler.base_lrs[0],
            "current_lr": trainer.agent.optimizer.param_groups[0]["lr"],
            "scheduler": type(scheduler).__name__, "T_max": scheduler.T_max,
            "eta_min": scheduler.eta_min,
            "scheduler_step_frequency": "once after each full 1024-sample buffer optimization",
            "models_module_path": str(Path(models.__file__).resolve()),
            "reference_path": trainer.metadata.get("reference_path"),
            "pretrained_sha256": trainer.metadata.get("pretrained_sha256")}


@dataclass
class MixedConfig:
    steps_14: int = 1000000
    steps_18: int = 850000
    restart_at: int = 1000000
    seed: int = 20261013
    buffer_size: int = 1024
    epochs: int = 10
    window_size: int = 500
    checkpoint_every: int = 10000
    smoke: bool = False

    def validate(self):
        if min(self.steps_14, self.steps_18, self.buffer_size, self.epochs,
               self.window_size, self.checkpoint_every) <= 0:
            raise ValueError("Budgets and training settings must be positive")
        if not 0 < self.restart_at < self.steps_14 + self.steps_18:
            raise ValueError("Restart must lie strictly inside the total budget")
        if not self.smoke and (self.steps_14, self.steps_18, self.restart_at,
                              self.buffer_size, self.epochs, self.window_size) != (
                                  1000000, 850000, 1000000, 1024, 10, 500):
            raise ValueError("Non-smoke runs require the exact paper budgets/configuration")


class MixedTrainer:
    def __init__(self, config, model, reference, expressions, output,
                 device="cpu", env=None, metadata=None):
        config.validate()
        self.config, self.model = config, model
        self.reference, self.expressions = reference, expressions
        self.output, self.device = Path(output), torch.device(device)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.mix_rng = random.Random(config.seed)
        self.targets = {14: config.steps_14, 18: config.steps_18}
        self.steps = {14: 0, 18: 0}
        self.total_steps, self.phase = 0, 1
        self.completed = {14: 0, 18: 0}
        self.wins = {14: 0, 18: 0}
        self.started = {14: 0, 18: 0}
        self.truncated = {14: 0, 18: 0}
        self.recent = {m: deque(maxlen=config.window_size) for m in (14, 18)}
        self.buffer_updates = {1: 0, 2: 0}
        self.adam_updates = {1: 0, 2: 0}
        self.mixed_buffers = 0
        self.discarded = []
        self.restarts = []
        self.best_18 = None
        self.best_18_step = None
        self.finished = False
        self.runtime_before_resume = 0.0
        self.clock_start = time.perf_counter()
        self.current_mines = self._sample_mines()
        self.env = env or CustomObservationWrapper(
            MinesweeperEnv(9, 9, self.current_mines, "rgb_array"), encoding_type="one_hot")
        self.env.unwrapped.num_mines = self.current_mines
        self.agent = reference.Agent(self.env, model, self.device, str(self.output),
                                     num_samples=config.buffer_size, epochs=config.epochs)
        if not isinstance(self.agent.optimizer, optim.Adam):
            raise ValueError("Mixed adaptation requires the reference Adam optimizer")
        if hasattr(self.agent, "scheduler") and not isinstance(self.agent.scheduler, lr_scheduler.CosineAnnealingLR):
            raise ValueError("Expected the reference CosineAnnealingLR or a disabled scheduler")
        self.original_update = self.agent.update_model
        self.agent.update_model = self._update_model
        self.buffer_mines = np.zeros(config.buffer_size, dtype=np.int32)
        self.started[self.current_mines] += 1
        self.episode_steps, self.episode_reward = 0, 0.0
        self.episode_active = True
        self._attach_optimizer_counter()
        self.model.train()  # retain original rollout Dropout behavior

    def _sample_mines(self):
        remaining = {m: self.targets[m] - self.steps[m] for m in (14, 18)}
        total = sum(remaining.values())
        if not total:
            return None
        return 14 if self.mix_rng.random() < remaining[14] / total else 18

    def _attach_optimizer_counter(self):
        def after_step(optimizer, args, kwargs):
            self.adam_updates[self.phase] += 1
        optimizer = self.agent.optimizer
        register = getattr(optimizer, "register_step_post_hook", None)
        if callable(register):
            self.optimizer_hook = register(after_step)
        else:
            # Older PyTorch: preserve scheduler's wrapped step and its attributes.
            original_step = optimizer.step
            @wraps(original_step)
            def counted_step(*args, **kwargs):
                result = original_step(*args, **kwargs)
                after_step(optimizer, args, kwargs)
                return result
            optimizer.step = counted_step
            class StepCounterHandle:
                def remove(self):
                    if optimizer.step is counted_step:
                        optimizer.step = original_step
            self.optimizer_hook = StepCounterHandle()

    def _event(self, kind, **fields):
        record = {"event": kind, "total_adaptation_steps": self.total_steps,
                  "steps_14": self.steps[14], "steps_18": self.steps[18],
                  "phase": self.phase, **fields}
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def _update_model(self):
        counts = dict(Counter(map(int, self.buffer_mines)))
        assert sum(counts.values()) == self.config.buffer_size and set(counts) <= {14, 18}
        before = self.adam_updates[self.phase]
        scheduler = getattr(self.agent, "scheduler", None)
        scheduler_before = scheduler.last_epoch if scheduler is not None else None
        augmentation_before = getattr(self.reference, "augmentation_calls", 0)
        self.original_update()  # exact original augmentation/DataLoader/loss/optimization
        if self.metadata.get("require_formal_configuration") and self.reference.augmentation_calls != augmentation_before + 1:
            raise RuntimeError("Expected exactly one paired augmentation per full buffer")
        updates = self.adam_updates[self.phase] - before
        expected = self.config.epochs * ((self.config.buffer_size + 31) // 32)
        if updates != expected:
            raise RuntimeError(f"Unexpected Adam update count: {updates} vs {expected}")
        if scheduler is not None and scheduler.last_epoch != scheduler_before + 1:
            raise RuntimeError("Reference scheduler must step once per full buffer")
        self.buffer_updates[self.phase] += 1
        self.mixed_buffers += int(counts.get(14, 0) > 0 and counts.get(18, 0) > 0)
        self._event("buffer_optimized", samples_by_mines=counts, adam_updates=updates,
                    learning_rate=self.agent.optimizer.param_groups[0]["lr"],
                    scheduler_last_epoch=scheduler.last_epoch if scheduler is not None else None)
        if sum(self.buffer_updates.values()) <= 2:
            print(json.dumps({"event": "buffer_optimized", "buffer": sum(self.buffer_updates.values()),
                              "adam_updates": updates, "samples_by_mines": counts,
                              "learning_rate": self.agent.optimizer.param_groups[0]["lr"],
                              "scheduler_last_epoch": scheduler.last_epoch if scheduler is not None else None}), flush=True)

    def _new_game(self):
        self.current_mines = self._sample_mines()
        if self.current_mines is None:
            return
        self.env.unwrapped.num_mines = self.current_mines
        self.agent.state = self.env.reset()
        self.started[self.current_mines] += 1
        self.episode_steps, self.episode_reward = 0, 0.0
        self.episode_active = True
        self._event("episode_started", mines=self.current_mines)

    def _save_weights(self, name):
        path = self.output / name
        temporary = self.output / (name + ".tmp")
        torch.save(self.model.state_dict(), temporary)
        os.replace(temporary, path)

    def _complete_game(self, info):
        mines = self.current_mines
        if info.get("status") not in ("win", "lose"):
            raise RuntimeError("Unexpected terminal status")
        won = int(info["status"] == "win")
        self.completed[mines] += 1
        self.wins[mines] += won
        self.recent[mines].append(won)
        self.episode_active = False
        self._event("episode_completed", mines=mines, status=info["status"],
                    episode_steps=self.episode_steps, reward=self.episode_reward)
        # Same checkpoint selection cadence as original AGG_STATS_EVERY=500.
        # Recent windows are maintained continuously; compare only full 18-mine blocks.
        if mines == 18 and self.completed[18] % self.config.window_size == 0:
            rate = sum(self.recent[18]) / self.config.window_size
            if self.best_18 is None or rate > self.best_18:
                self.best_18, self.best_18_step = rate, self.total_steps
                self._save_weights("mixed_best_18_online.dat")
                self._event("best_18_saved", win_rate=rate, file="mixed_best_18_online.dat")

    def _discard_buffer(self, reason):
        count = self.agent.current_sample
        record = {"reason": reason, "step": self.total_steps, "samples": count,
                  "samples_by_mines": dict(Counter(map(int, self.buffer_mines[:count])))}
        self.discarded.append(record)
        self.agent.current_sample = 0
        self.buffer_mines[:] = 0
        self._event("buffer_discarded", **record)

    def _restart_optimizer(self):
        self._discard_buffer("optimizer_boundary")
        self.optimizer_hook.remove()
        self.agent.optimizer = eval(self.expressions["optimizer"], self.reference.__dict__, {"self": self.agent})
        if "scheduler" in self.expressions:
            self.agent.scheduler = eval(self.expressions["scheduler"], self.reference.__dict__, {"self": self.agent})
        self.phase = 2
        self.restarts.append(self.total_steps)
        self._attach_optimizer_counter()
        self._event("optimizer_restarted", learning_rate=self.agent.optimizer.param_groups[0]["lr"],
                    current_episode_mines=self.current_mines)

    def step(self):
        if self.finished:
            return False
        mines = self.current_mines
        assert self.steps[mines] < self.targets[mines]
        state = self.agent.state
        # No true-board read in action selection; training mode remains enabled.
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            scores = self.model(tensor).cpu().numpy().squeeze().copy()
        scores[state[-1].reshape(-1) == 0] = np.inf
        action = int(np.argmin(scores))
        if not np.isfinite(scores[action]) or state[-1].reshape(-1)[action] != 1:
            raise RuntimeError("No legal unknown action in an unfinished game")
        self.env.button = "left"
        next_state, reward, done, info = self.env.step(action)
        self.steps[mines] += 1
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += reward
        self.agent.state = np.array(next_state, copy=True)
        # Full truth is accessed ONLY after the action, for the original label.
        label = self.reference._convert_board(self.env.board)
        if int(label.sum()) != mines or self.env.unwrapped.num_mines != mines:
            raise RuntimeError("Current environment/label density mismatch")
        index = self.agent.current_sample
        self.buffer_mines[index] = mines
        self.agent.increment_sample_size(self.agent.state, label)
        if done:
            self._complete_game(info)
        elif self.steps[mines] == self.targets[mines]:
            self.truncated[mines] += 1
            self.episode_active = False
            self._event("episode_budget_truncated", mines=mines, episode_steps=self.episode_steps,
                        reward=self.episode_reward)
        if self.total_steps == self.config.restart_at:
            self._restart_optimizer()  # retain model weights and current game/difficulty
        if self.total_steps == sum(self.targets.values()):
            self._discard_buffer("final_budget")  # original stop leaves partial buffer untrained
            self.finished = True
            assert self.steps == self.targets
            assert all(self.started[m] == self.completed[m] + self.truncated[m] for m in (14, 18))
            self._save_weights("mixed_final.dat")
            self._event("training_completed")
            self.save_recovery()
            self.write_summary()
        else:
            if not self.episode_active:
                self._new_game()
            if self.total_steps % self.config.checkpoint_every == 0 or self.total_steps == self.config.restart_at:
                self.save_recovery()
                self.write_summary()
        return True

    def summary(self):
        return {"finished": self.finished, "steps_14": self.steps[14], "steps_18": self.steps[18],
                "total_adaptation_steps": self.total_steps, "phase": self.phase,
                "games_completed": self.completed, "wins": self.wins,
                "online_win_rate": {m: self.wins[m] / self.completed[m] if self.completed[m] else None for m in (14, 18)},
                "recent_win_rate": {m: sum(self.recent[m]) / len(self.recent[m]) if self.recent[m] else None for m in (14, 18)},
                "recent_window_lengths": {m: len(self.recent[m]) for m in (14, 18)},
                "games_started": self.started, "budget_truncated": self.truncated,
                "buffer_optimizations_by_phase": self.buffer_updates,
                "adam_updates_by_phase": self.adam_updates,
                "mixed_buffers": self.mixed_buffers, "pending_samples": self.agent.current_sample,
                "discarded_samples": self.discarded, "optimizer_restart_steps": self.restarts,
                "learning_rate": self.agent.optimizer.param_groups[0]["lr"],
                "best_18_online_win_rate": self.best_18, "best_18_step": self.best_18_step,
                "best_18_checkpoint": "mixed_best_18_online.dat" if self.best_18 is not None else None,
                "elapsed_seconds": self.runtime_before_resume + time.perf_counter() - self.clock_start}

    def write_summary(self):
        (self.output / "summary.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    def save_recovery(self):
        base = self.env.unwrapped
        log_path = self.output / "events.jsonl"
        state = {"config": asdict(self.config), "metadata": self.metadata,
                 "model": self.model.state_dict(), "optimizer": self.agent.optimizer.state_dict(),
                 "scheduler": self.agent.scheduler.state_dict() if hasattr(self.agent, "scheduler") else None,
                 "counters": {name: copy.deepcopy(getattr(self, name)) for name in (
                     "steps", "total_steps", "phase", "completed", "wins", "started", "truncated",
                     "buffer_updates", "adam_updates", "mixed_buffers", "discarded", "restarts",
                     "best_18", "best_18_step", "finished", "current_mines", "episode_steps",
                     "episode_reward", "episode_active")},
                 "recent": {m: list(self.recent[m]) for m in (14, 18)},
                 "pending_samples": self.agent.current_sample,
                 "features": self.agent.features[:self.agent.current_sample].copy(),
                 "labels": self.agent.labels[:self.agent.current_sample].copy(),
                 "buffer_mines": self.buffer_mines.copy(), "observation": self.agent.state.copy(),
                 "environment": {name: copy.deepcopy(getattr(base, name)) for name in (
                     "num_mines", "board", "player_board", "reward", "done", "info", "first_move", "stats", "button")},
                 "mix_rng": self.mix_rng.getstate(), "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
                 "elapsed_seconds": self.summary()["elapsed_seconds"],
                 "event_log_bytes": log_path.stat().st_size if log_path.exists() else 0,
                 "best_weights": torch.load(self.output / "mixed_best_18_online.dat", map_location="cpu", weights_only=True)
                     if self.best_18 is not None else None}
        temporary = self.output / "mixed_resume.pt.tmp"
        torch.save(state, temporary)
        os.replace(temporary, self.output / "mixed_resume.pt")

    def restore(self, saved):
        if saved["config"] != asdict(self.config):
            raise ValueError("Resume configuration mismatch")
        log_path = self.output / "events.jsonl"
        offset = saved["event_log_bytes"]
        if offset and (not log_path.exists() or log_path.stat().st_size < offset):
            raise ValueError("Resume requires the original events.jsonl alongside the checkpoint")
        if log_path.exists():
            with log_path.open("r+b") as stream:
                stream.truncate(offset)  # discard logs after the last consistent checkpoint
        self.model.load_state_dict(saved["model"], strict=True)
        for name, value in saved["counters"].items():
            setattr(self, name, value)
        self.agent.optimizer.load_state_dict(saved["optimizer"])
        if saved["scheduler"] is not None:
            self.agent.scheduler.load_state_dict(saved["scheduler"])
        self.recent = {m: deque(saved["recent"][m], maxlen=self.config.window_size) for m in (14, 18)}
        self.agent.current_sample = saved["pending_samples"]
        self.agent.features[:self.agent.current_sample] = saved["features"]
        self.agent.labels[:self.agent.current_sample] = saved["labels"]
        self.buffer_mines[:] = saved["buffer_mines"]
        self.agent.state = saved["observation"].copy()
        for name, value in saved["environment"].items():
            setattr(self.env.unwrapped, name, copy.deepcopy(value))
        self.mix_rng.setstate(saved["mix_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] is not None:
            if self.device.type != "cuda":
                raise ValueError("Resume must use the original CUDA device type")
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        self.runtime_before_resume, self.clock_start = saved["elapsed_seconds"], time.perf_counter()
        if saved["best_weights"] is not None:
            path = self.output / "mixed_best_18_online.dat"
            torch.save(saved["best_weights"], path)
        else:
            (self.output / "mixed_best_18_online.dat").unlink(missing_ok=True)
        if self.finished:
            self._save_weights("mixed_final.dat")
        else:
            (self.output / "mixed_final.dat").unlink(missing_ok=True)
        self._event("resumed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, help="Exact common 10-mine state_dict file, not a directory")
    parser.add_argument("--reference", type=Path, default=ROOT / "minesweeper_train_cnn.py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Synthetic weights; small budgets/window; NOT a paper run")
    parser.add_argument("--validate-buffers", type=int, help="Real-checkpoint preflight: stop safely after N full buffers")
    args = parser.parse_args(argv)
    if args.resume and args.smoke:
        parser.error("Smoke configuration is recovered automatically; do not combine --resume and --smoke")
    if args.validate_buffers is not None and (args.validate_buffers < 2 or args.smoke or args.resume):
        parser.error("--validate-buffers requires >=2 and a new real-checkpoint run")
    saved = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config = (MixedConfig(**saved["config"]) if saved else MixedConfig(seed=args.seed))
    if args.smoke:
        config = MixedConfig(steps_14=1079, steps_18=1077, restart_at=1075,
                             seed=args.seed, window_size=5, checkpoint_every=1000, smoke=True)
    config.validate()
    if not saved and not args.smoke and (args.pretrained is None or not args.pretrained.is_file()):
        parser.error("Supply the exact shared pretrained .dat file with --pretrained")
    output = args.output or (args.resume.resolve().parent if saved else ROOT / "output/mixed_adaptation" / str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=True)
    if not saved and (output / "metadata.json").exists():
        raise FileExistsError("Choose a new Mixed output directory")
    source_hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in (
        ROOT / "lib/models.py", ROOT / "lib/wrappers.py", ROOT / "minesweeper_pygame.py",
        ROOT / "constants.py", Path(__file__).resolve())}
    source_hashes["reference_training"] = file_hash(args.reference)
    if saved and source_hashes != saved["metadata"]["source_sha256"]:
        raise ValueError("Source changed since the resume checkpoint; restore the saved configuration first")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose --device cpu")
        torch.cuda.manual_seed_all(config.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    reference, expressions, source, augmentation = load_reference(args.reference, output)
    original_augmentation = augmentation
    augmentation = prepare_mixed_reference(reference, augmentation)
    model = models.Model((10, 9, 9), 81).to(args.device)
    configure_mixed_loss(model)
    if not saved and not args.smoke:
        weights = torch.load(args.pretrained, map_location=args.device, weights_only=True)
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as exc:
            raise RuntimeError("Pretrained architecture does not match current lib/models.py. Restore the original "
                               "SCL model/Dropout layer layout yourself; Mixed never remaps or changes weights.") from exc
    metadata = (saved["metadata"] if saved else {
        "config": asdict(config), "pretrained_path": str(args.pretrained.resolve()) if args.pretrained else None,
        "pretrained_sha256": file_hash(args.pretrained) if args.pretrained else None,
        "initial_weights": "synthetic_random_smoke_only" if args.smoke else "explicit_shared_pretrained_file",
        "pretraining_interactions": None if args.smoke else "approximately 6000000, user provenance; not inferable from weights",
        "reference_path": str(args.reference.resolve()), "source_sha256": source_hashes,
        "loss": type(model.loss).__name__, "dropout": [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)],
        "augmentation_enabled": augmentation, "batch_size": 32,
        "reference_augmentation_enabled": original_augmentation,
        "augmentation_configuration": "Mixed startup wrapper reusing reference augment_data" if not original_augmentation else "reference update_model",
        "require_formal_configuration": True,
        "run_role": "synthetic_smoke" if args.smoke else "real_checkpoint_preflight" if args.validate_buffers else "formal_adaptation",
        "best_selection": "18-mines-only full recent window, every 500 completed 18-mines games (5 in smoke)",
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "device": args.device, "deterministic_algorithms": True})
    if saved and args.device != metadata["device"]:
        raise ValueError("Resume must use the original device type")
    trainer = MixedTrainer(config, model, reference, expressions, output, args.device, metadata=metadata)
    startup = validate_startup(trainer, augmentation)
    metadata["startup"] = startup
    print("MIXED_STARTUP " + json.dumps(startup, default=str), flush=True)
    if saved:
        trainer.restore(saved)
    else:
        metadata["optimizer_defaults"] = trainer.agent.optimizer.defaults
        scheduler = getattr(trainer.agent, "scheduler", None)
        metadata["scheduler"] = scheduler.state_dict() if scheduler is not None else None
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        (output / "reference_training_snapshot.py").write_text(source, encoding="utf-8")
        (output / "reference_model_snapshot.py").write_text((ROOT / "lib/models.py").read_text(encoding="utf-8"), encoding="utf-8")
        trainer._event("experiment_started", mines=trainer.current_mines)
        trainer.save_recovery()  # recoverable even before the first periodic checkpoint
    print(f"Mixed output: {output.resolve()}; loss={metadata['loss']}; "
          f"augmentation={metadata['augmentation_enabled']}; scheduler={type(getattr(trainer.agent, 'scheduler', None)).__name__}", flush=True)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        while trainer.step():
            if args.validate_buffers and sum(trainer.buffer_updates.values()) >= args.validate_buffers:
                trainer._event("real_checkpoint_preflight_completed", buffers=sum(trainer.buffer_updates.values()))
                trainer.save_recovery()
                trainer.write_summary()
                print("Real-checkpoint preflight complete; full adaptation budget has not been run.", flush=True)
                break
            if stop_requested and not trainer.finished:
                # Finish the current action/full-buffer optimization before saving.
                trainer._event("graceful_stop")
                trainer.save_recovery()
                trainer.write_summary()
                print("Stopped at a consistent action boundary; recovery checkpoint saved.", flush=True)
                break
            if trainer.total_steps % config.checkpoint_every == 0:
                print(json.dumps(trainer.summary()), flush=True)
    except KeyboardInterrupt:
        # Interruption at an optimization boundary can be partial: preserve only
        # the last fully consistent recovery checkpoint, not an in-flight batch.
        print("Interrupted. Resume from the last atomic mixed_resume.pt; in-flight work is not certified.", flush=True)
        return
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        trainer.env.close()
    print(json.dumps(trainer.summary(), indent=2), flush=True)


if __name__ == "__main__":
    main()
