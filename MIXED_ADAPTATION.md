# Mixed Adaptation

Entry: `minesweeper_train_mixed.py`. It reuses the current training script's
Agent/update_model, augmentation, Adam arguments and optional scheduler. The
original training script and environment are unchanged. Dropout(0.3) was restored
in `lib/models.py` at the user's request; the common checkpoint loads strictly.

Mixed startup explicitly sets unweighted mean BCEWithLogitsLoss on its model
instance, requires Dropout=[0.3,0.3] and an active reference CosineAnnealingLR,
and enables paired augmentation using the reference augment_data function.
If reference augmentation is off, only the dynamically loaded Mixed Agent is
wrapped; the original file and Fixed model defaults are not changed. Actual
reference Adam lr=1e-5, T_max=5000, eta_min=1e-7; scheduler.step runs once per
full buffer. Startup details and the first two optimized buffers are printed.

```bash
python minesweeper_train_mixed.py --pretrained models/CNN-1763950674/Minesweeper-v0-best_win_rate.dat --device cuda
python minesweeper_train_mixed.py --smoke --device cuda
python minesweeper_train_mixed.py --pretrained models/CNN-1763950674/Minesweeper-v0-best_win_rate.dat --validate-buffers 2 --device cuda
python minesweeper_train_mixed.py --resume output/mixed_adaptation/YOUR_RUN/mixed_resume.pt --device cuda
python -m unittest discover -s tests -p test_mixed_adaptation.py -v
```

Choose the exact file shared by SCL/Direct Transfer (best_reward and best_win_rate
are different files). New runs get independent output directories. Resume requires
the original run directory and unchanged sources/device type. Ctrl+C finishes the
current action/optimization and saves a consistent recovery checkpoint; hard
crashes roll back to the last checkpoint and may replay uncommitted work.

Each new game samples difficulty proportional to remaining per-difficulty budgets.
Difficulty stays fixed within a game. One shared 1024-sample buffer collects
post-action observations and original true-mine labels. Quota-truncated games
are excluded from completed-game wins/losses. No truth is read for action choice.

Full budget: 14 mines=1,000,000; 18 mines=850,000; total=1,850,000 actions.
At total action 1,000,000, discard the partial buffer and restart Adam and the
enabled scheduler, preserving weights and the current game. At the final budget,
discard the last partial buffer. Expected full-buffer optimizations: 976 + 830;
Adam updates: 312,320 + 265,600; discarded samples: 576 + 80. These are arithmetic
expectations, not completed full-experiment results.

`mixed_best_18_online.dat` is selected using complete 500-game 18-mine windows at
the original 500-game evaluation cadence. Both recent windows are maintained
continuously. `mixed_final.dat` is separate; `mixed_resume.pt` is overwritten
atomically. Metadata, JSONL events and summary record configuration, hashes,
budgets, wins, truncations, optimizer updates, learning rate and runtime.

Smoke mode uses synthetic initial weights, 1079/1077 actions, restart at 1075,
and a 5-game window; it retains 1024 samples, 10 epochs and batch size 32.
It never substitutes for a real-checkpoint test. --validate-buffers loads the
explicit pretrained file with strict=True, keeps the formal budgets/500-game
window, and stops safely after the requested number of full optimizations.
Seven unit/integration tests passed, including old-PyTorch step-counter compatibility,
real-model startup guards and scheduler restart in a temporary
reference fixture, exact quotas, shared mixed buffers, label/augmentation alignment,
and bit-exact interrupted/resumed CPU training. Validation artifacts are cleaned
after inspection. Real-checkpoint preflight completed 2048 actions / two full
buffers / 640 Adam updates. Startup and first-buffer evidence is recorded in
`output/mixed_adaptation/prelaunch_checks.json`.

CUDA smoke with restored Dropout passed: 2156 actions (1079/1077), two mixed
buffer optimizations, 640 Adam updates, one restart at 1075, and 51+57
discarded samples. Best/final/recovery files were verified before cleanup.
