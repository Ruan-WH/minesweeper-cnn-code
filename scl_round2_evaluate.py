"""Run the unchanged original 10x1000 statistics logic with a supplied checkpoint.

Only model_path and environment mine count are substituted in its parsed AST.
This file never runs automatically during training or checkpoint selection.
"""
import argparse
import ast
import contextlib
import json
import os
from pathlib import Path
import random
import sys

from scl_round2 import ROOT, digest, write_json, now


def prepare_source(source, model_path):
    tree = ast.parse(source)
    changed_model = changed_mines = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model_path" for t in node.targets):
            node.value = ast.Constant(str(model_path))
            changed_model += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MinesweeperEnv":
            if len(node.args) != 4:
                raise ValueError("Unexpected original evaluation environment arguments")
            node.args[2] = ast.Constant(18)
            changed_mines += 1
    if (changed_model, changed_mines) != (1, 1):
        raise ValueError("Original evaluation interface changed; manual review required")
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if not args.gpu.isdigit() or not 0 <= args.seed < 2**32:
        parser.error("One GPU index and uint32 seed required")
    model = Path(args.model).resolve(strict=True)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    import numpy as np
    import torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    path = ROOT / "minesweeper_test_cnn_statistics.py"
    source = path.read_text(encoding="utf-8-sig")
    tree = prepare_source(source, model)
    namespace = {"__name__": "__main__", "__file__": str(path)}
    report = {"status": "running", "started": now(), "model": str(model),
              "model_sha256": digest(model), "source_sha256": digest(path), "seed": args.seed,
              "protocol": "Original statistics loop; RNG seeded once before script; fixed first action 0; eval mode; safe first click; 10 groups x1000"}
    write_json(out / "result.json", report)
    try:
        with open(out / "console.log", "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            exec(compile(tree, str(path), "exec"), namespace)
        if namespace["num_groups"] != 10 or namespace["episodes_per_group"] != 1000:
            raise RuntimeError("Original evaluation group counts changed")
        if namespace["net"].training or namespace["first_move"]:
            raise RuntimeError("Evaluation mode/first-move state invalid")
        rates = namespace["win_rates"]
        if len(rates) != 10:
            raise RuntimeError("Incomplete evaluation")
        report.update(status="completed", ended=now(), games=10000,
                      group_win_rates=rates, win_rate=float(np.mean(rates)),
                      group_std_population=float(np.std(rates)),
                      wins=sum(round(r * 1000) for r in rates))
        write_json(out / "result.json", report)
        print(json.dumps(report))
    except BaseException as exc:
        report.update(status="failed", ended=now(), error=repr(exc))
        write_json(out / "result.json", report)
        raise


if __name__ == "__main__":
    main()
