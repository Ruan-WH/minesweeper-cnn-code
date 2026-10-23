"""Sequential SCL replication. Original training/evaluation files are never edited.

python scl_round2.py --config configs/scl_round2.json --output output/NEW_RUN
Runs a separate six-stage smoke chain first, then the six formal stages.
No resume: a failed chain must be investigated and restarted in a fresh directory.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SOURCES = ["scl_round2.py", "minesweeper_train_cnn.py", "minesweeper_train_mixed.py",
           "lib/models.py", "lib/wrappers.py", "minesweeper_pygame.py",
           "constants.py", "init_pygame.py", "minesweeper_test_cnn_statistics.py"]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_config(c):
    if len(c["seeds"]) != 2 or len(set(c["seeds"])) != 2:
        raise ValueError("Two distinct seeds required")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in c["seeds"]):
        raise ValueError("Seeds must be uint32 integers")
    if c["stages"] != [{"mines": 10, "budget": 6000000},
                       {"mines": 14, "budget": 1000000},
                       {"mines": 18, "budget": 850000}]:
        raise ValueError("Formal stage budgets must match the paper request")
    if (c["window"], c["augmentation"], c["selection"], c["learning_rate"]) != (500, True, "rolling", 1e-4):
        raise ValueError("Configuration must match the user-confirmed protocol")
    if c["min_free_gib"] < 2 or c["progress_every"] < 1:
        raise ValueError("At least 2 GiB reserve and positive progress interval required")
    if not str(c["gpu"]).isdigit():
        raise ValueError("Select exactly one physical GPU index")


def check_disk(path, reserve):
    free = shutil.disk_usage(path).free
    if free < reserve * 1024**3:
        raise RuntimeError("Insufficient disk space: %.2f GiB free" % (free / 1024**3))
    return free


def event(destination, **values):
    with open(destination, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), **values}, allow_nan=False) + "\n")


def load_weights(torch, path, device):
    # Load only plain model state dictionaries, never optimizer/pickle state.
    return torch.load(path, map_location=device, weights_only=True)


def verify_model(torch, model, path, shape):
    if not path.is_file():
        raise RuntimeError("Missing best checkpoint: " + str(path))
    expected = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    if not expected <= path.stat().st_size <= expected * 1.2 + 1024**2:
        raise RuntimeError("Checkpoint size inconsistent with model tensors")
    # No new model construction / random draws in training RNG streams.
    state = load_weights(torch, path, "cpu")
    import copy
    probe = copy.deepcopy(model).cpu()
    probe.load_state_dict(state, strict=True)
    probe.eval()
    with torch.no_grad():
        result = probe(torch.zeros(1, *shape))
    if result.shape != (1, 81) or not torch.isfinite(result).all():
        raise RuntimeError("Checkpoint forward validation failed")
    return digest(path)


def worker(job):
    # Imports occur only after the parent has constrained CUDA visibility.
    import random
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from lib import models
    from lib.wrappers import CustomObservationWrapper
    from minesweeper_pygame import MinesweeperEnv
    from minesweeper_train_mixed import load_reference, prepare_mixed_reference, validate_startup

    out = Path(job["output"])
    config = job["config"]
    stage = job["stage"]
    step = 0
    started = time.monotonic()
    report = {"status": "running", "started": now(), "seed": job["seed"],
              "mines": stage["mines"], "budget": stage["budget"],
              "smoke": job["smoke"], "actual_steps": 0, "best": None}
    write_json(out / "result.json", report)
    writer = env = None
    loss_count = 0
    actual_steps = 0
    losses = []
    try:
        for name, expected in job["source_hashes"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError("Source changed during run: " + name)
        if job["device"] == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("Exactly one visible CUDA GPU required")
        torch.set_num_threads(1)
        seed = job["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # As in the reference: no forced deterministic kernels or LR changes.
        reference, _, _, active_augmentation = load_reference(ROOT / "minesweeper_train_cnn.py", out)
        if reference.LEARNING_RATE != config["learning_rate"]:
            raise RuntimeError("Reference LR no longer matches confirmed configuration")
        prepare_mixed_reference(reference, active_augmentation)
        env = CustomObservationWrapper(MinesweeperEnv(9, 9, stage["mines"], "rgb_array"), "one_hot")
        raw_step = env.env.step
        def counted_step(action):
            nonlocal actual_steps
            value = raw_step(action)
            actual_steps += 1
            return value
        env.env.step = counted_step
        # Board generation/reset use global NumPy RNG; no independent board RNG.
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        device = torch.device(job["device"])
        model = models.Model(env.observation_space.shape, env.action_space.n).to(device)
        initial = job.get("initial")
        if initial:
            if digest(initial["path"]) != initial["sha256"]:
                raise RuntimeError("Upstream best weights changed")
            model.load_state_dict(load_weights(torch, initial["path"], device), strict=True)
        model.train()
        agent = reference.Agent(env, model, device, str(out))
        # Validate the original Agent's optimizer/scheduler/loss/dropout at runtime.
        from types import SimpleNamespace
        startup = validate_startup(SimpleNamespace(model=model, agent=agent,
                                  metadata={"reference_path": str(ROOT / "minesweeper_train_cnn.py")}), True)
        if startup["initial_lr"] != 1e-4 or startup["T_max"] != 5000 or startup["eta_min"] != 1e-7:
            raise RuntimeError("Unexpected learning-rate protocol")
        metadata = {**job, "runtime": startup, "python": sys.version,
                    "torch": torch.__version__, "numpy": np.__version__,
                    "gpu_name": torch.cuda.get_device_name(0) if job["device"] == "cuda" else None,
                    "model": repr(model), "parameter_count": sum(p.numel() for p in model.parameters()),
                    "buffer": {"samples": 1024, "epochs": 10, "batch_size": 32, "shuffle": True},
                    "rng": "seed Python/NumPy/Torch CPU and CUDA before environment/model; same seed reset per stage; boards use global NumPy",
                    "augmentation": "one paired transform of whole full buffer, reference augment_data",
                    "checkpoint_rule": "strict improvement, full recent window, checked after every completed episode",
                    "partial_buffer": "no partial-buffer optimization; exact step budget may end within a game"}
        write_json(out / "config.json", metadata)
        writer = SummaryWriter(log_dir=str(out / "tensorboard"), max_queue=100, flush_secs=30)
        original_loss = model.compute_loss
        def tracked_loss(*args, **kwargs):
            nonlocal loss_count
            loss = original_loss(*args, **kwargs)
            if not torch.isfinite(loss).all():
                raise RuntimeError("Nonfinite training loss")
            # Keep logging separate from the original update implementation.
            losses.append(float(loss.detach().item()))
            loss_count += 1
            return loss
        model.compute_loss = tracked_loss
        recent = deque(maxlen=job["window"])
        rewards, moves, cells, guesses = [], [], [], []
        best_rate = -1.0
        best_path = out / "best_win_rate.dat"
        progress = out / "events.jsonl"
        last_time = time.monotonic()
        last_step = 0
        episode_time = time.monotonic()
        episode_step = 0
        for step in range(1, stage["budget"] + 1):
            reward, move, cell, guess, games, _, stats = agent.play_step()
            if reward is not None:
                episode_now = time.monotonic()
                writer.add_scalar("speed", (step - episode_step) / max(episode_now - episode_time, 1e-9), step)
                episode_time, episode_step = episode_now, step
                recent.append(int(stats["n_win"]))
                rewards.append(reward); moves.append(move); cells.append(cell); guesses.append(guess)
                for tag, value in (("episode_reward", reward), ("moves_made", move), ("cells_left", cell)):
                    writer.add_scalar(tag, value, step)
                if len(recent) == job["window"]:
                    rate = sum(recent) / len(recent)
                    writer.add_scalar("rolling_win_rate", rate, step)
                    if rate > best_rate:
                        check_disk(out, config["min_free_gib"])
                        tmp = out / "best_win_rate.dat.tmp"
                        with open(tmp, "wb") as f:
                            torch.save(model.state_dict(), f)
                            f.flush(); os.fsync(f.fileno())
                        # Validate before replacing the previous best.
                        checkpoint_hash = verify_model(torch, model, tmp, env.observation_space.shape)
                        os.replace(tmp, best_path)
                        best_rate = rate
                        report["best"] = {"step": step, "rolling_win_rate": rate,
                                          "learning_rate": agent.optimizer.param_groups[0]["lr"],
                                          "path": str(best_path), "sha256": checkpoint_hash}
                        write_json(out / "best.json", report["best"])
                        event(progress, type="best", **report["best"])
                # Preserve existing per-episode TensorBoard tags and stats.txt schema.
                if games % 500 == 0:
                    writer.add_scalar("mean_episode_reward", float(np.mean(rewards)), step)
                    writer.add_scalar("win_rate", agent.games_won / 500, step)
                    agent.save_stats(games, agent.games_won, float(np.mean(moves)), float(np.mean(cells)), float(np.mean(guesses)))
                    agent.games_won = 0
                    rewards.clear(); moves.clear(); cells.clear(); guesses.clear()
            if step % config["progress_every"] == 0 or step == stage["budget"]:
                elapsed = time.monotonic()
                values = {"actual_steps": step, "games": agent.games_played,
                          "rolling_win_rate": sum(recent) / len(recent) if recent else None,
                          "loss": sum(losses) / len(losses) if losses else None,
                          "learning_rate": agent.optimizer.param_groups[0]["lr"],
                          "buffer_updates": step // 1024, "adam_updates": loss_count,
                          "augmentation_calls": reference.augmentation_calls,
                          "pending_buffer_samples": agent.current_sample,
                          "speed": (step - last_step) / max(elapsed - last_time, 1e-9)}
                for tag in ("learning_rate", "loss", "speed"):
                    if values[tag] is not None:
                        writer.add_scalar(tag, values[tag], step)
                event(progress, type="progress", **values)
                report.update(values)
                write_json(out / "result.json", report)
                print(json.dumps(values), flush=True)
                losses.clear(); last_time = elapsed; last_step = step
                check_disk(out, config["min_free_gib"])
        if report["best"] is None:
            raise RuntimeError("No full checkpoint-selection window completed; stage fails")
        if actual_steps != stage["budget"]:
            raise RuntimeError("Actual environment step count differs from budget")
        if reference.augmentation_calls != step // 1024 or loss_count != (step // 1024) * 320:
            raise RuntimeError("Buffer/augmentation/Adam update counts do not match protocol")
        if agent.scheduler.last_epoch != step // 1024:
            raise RuntimeError("Scheduler update count mismatch")
        final_hash = verify_model(torch, model, best_path, env.observation_space.shape)
        if final_hash != report["best"]["sha256"]:
            raise RuntimeError("Best checkpoint hash mismatch")
        report.update(status="completed", actual_steps=step, ended=now(),
                      elapsed_seconds=time.monotonic() - started,
                      partial_buffer_samples_discarded=agent.current_sample,
                      unfinished_game_moves=agent.moves_made,
                      checkpoint_verified=True)
        write_json(out / "result.json", report)
    except BaseException:
        report.update(status="failed", ended=now(), actual_steps=actual_steps,
                      elapsed_seconds=time.monotonic() - started, error=traceback.format_exc())
        write_json(out / "result.json", report)
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def run_chain(root, config, hashes, smoke, device, envvars, results):
    label = "smoke" if smoke else "formal"
    for seed in config["seeds"]:
        initial = None  # Never transfer weights between seeds or from smoke to formal.
        for stage in config["stages"]:
            out = root / label / str(seed) / str(stage["mines"])
            out.mkdir(parents=True, exist_ok=False)
            check_disk(out, config["min_free_gib"])
            job = {"output": str(out), "seed": seed, "stage": {**stage, "budget": 1100} if smoke else stage,
                   "window": 5 if smoke else config["window"], "smoke": smoke,
                   "initial": initial, "config": config, "source_hashes": hashes, "device": device}
            job_path = out / "job.json"
            write_json(job_path, job)
            print("Starting", label, seed, stage["mines"], flush=True)
            # Blocking child: exactly one worker at a time; no shell or detached child.
            with open(out / "console.log", "w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                            "--worker", str(job_path)], cwd=ROOT,
                                           env=envvars, stdout=log, stderr=subprocess.STDOUT)
                write_json(root / "active_worker.json", {"pid": process.pid, "stage": str(out), "started": now()})
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise
            result_path = out / "result.json"
            report = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"status": "failed", "error": "worker did not write result"}
            if code != 0 or report["status"] != "completed":
                report["status"] = "failed"
                report["returncode"] = code
                report["console_log"] = str(out / "console.log")
                write_json(result_path, report)
                raise RuntimeError("Stage failed; no dependent stages started: " + str(out))
            if (report.get("actual_steps") != job["stage"]["budget"] or not report.get("checkpoint_verified")):
                raise RuntimeError("Invalid completion record: " + str(out))
            initial = {"path": report["best"]["path"], "sha256": report["best"]["sha256"]}
            if digest(initial["path"]) != initial["sha256"]:
                raise RuntimeError("Checkpoint changed after validation")
            results.append(report)
            write_json(root / "summary.json", results)
            print("Completed", label, seed, stage["mines"], "best:", initial["path"], flush=True)


def orchestrate(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.device != "cuda" and not args.smoke_only:
        raise ValueError("Formal training requires CUDA")
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError("Output must be a NEW directory: " + str(root))
    root.parent.mkdir(parents=True, exist_ok=True)
    check_disk(root.parent, config["min_free_gib"])
    # OS-held advisory lock; released on crash. Cross-platform, same project/GPU.
    lock = open(ROOT / (".scl_round2_gpu%s.lock" % config["gpu"]), "a+b")
    if os.name == "nt":
        import msvcrt
        lock.seek(0); lock.write(b"0"); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    results = []
    root.mkdir()
    state = {"status": "running", "pid": os.getpid(), "started": now(), "config": config,
             "output": str(root), "smoke_only": args.smoke_only}
    write_json(root / "run.json", state)
    try:
        hashes = {name: digest(ROOT / name) for name in SOURCES}
        for name in SOURCES:
            target = root / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write_json(root / "source_hashes.json", hashes)
        write_json(root / "config.json", config)
        envvars = dict(os.environ, CUDA_VISIBLE_DEVICES=config["gpu"], PYTHONHASHSEED="0",
                       SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1", TF_CPP_MIN_LOG_LEVEL="2")
        run_chain(root, config, hashes, True, args.device, envvars, results)
        write_json(root / "smoke_passed.json", {"passed": True, "ended": now(), "stages": 6,
                   "note": "1100 steps/stage, full 1024 buffer, original 10 epochs; selection window=5 FOR SMOKE ONLY"})
        if not args.smoke_only:
            run_chain(root, config, hashes, False, args.device, envvars, results)
            final = [r["best"]["path"] for r in results if not r["smoke"] and r["mines"] == 18]
            write_json(root / "final_checkpoints.json", {"checkpoints": final, "independent_test_per_checkpoint": 10000,
                       "evaluation": "Use unchanged minesweeper_test_cnn_statistics.py via a parameter wrapper; eval mode, action 0 first, original safe-first-click. No final testing performed here."})
        state.update(status="completed", ended=now())
    except BaseException:
        state.update(status="failed", ended=now(), error=traceback.format_exc())
        (root / "error.log").write_text(state["error"], encoding="utf-8")
        raise
    finally:
        write_json(root / "run.json", state)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/scl_round2.json"))
    parser.add_argument("--output")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt("Termination signal %s" % signum)
    signal.signal(signal.SIGTERM, terminate)
    if args.worker:
        worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
    elif not args.output:
        parser.error("--output must name a fresh experiment directory")
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
