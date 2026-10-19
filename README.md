# Minesweeper CNN

Minesweeper environment, CNN training and evaluation, mixed adaptation, and a CSP baseline.

## Project files

- `minesweeper_pygame.py`: Minesweeper environment.
- `minesweeper_train_cnn.py`, `minesweeper_train_mixed.py`: training entry points.
- `minesweeper_test_cnn.py`, `minesweeper_test_cnn_statistics.py`: CNN evaluation.
- `minesweeper_test_csp_statistics.py`: CSP baseline evaluation.
- `lib/`: models, wrappers, and CSP adapter.
- `tests/`: automated tests.
- `assets/`: game images and font.
- `output/`: recorded experiment results, reports, and source snapshots.

See [CSP_BASELINE.md](CSP_BASELINE.md), [CSP_RESULTS.md](CSP_RESULTS.md), and [MIXED_ADAPTATION.md](MIXED_ADAPTATION.md) for setup, commands, and experiment details. CSP dependencies are listed in `requirements-csp-baseline.txt`.

## Upload scope

This repository is a source snapshot of the local project on 2026-10-19. Training event logs, model weights/checkpoints, Python caches, and IDE configuration are excluded. Scripts referring to excluded checkpoints require those files to be supplied separately. The previous repository's commit history is not included.

Third-party CSP code and its original license are preserved in `vendor/nabenabe_csp/`.
