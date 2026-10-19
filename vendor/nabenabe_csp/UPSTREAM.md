# Vendored probability core

Source: https://github.com/nabenabe0928/mine-sweeper-solver
Commit: `3a499ddee44bf7562808c8ab787bd173fb68fa2e`
License: Apache-2.0; original LICENSE included without changes.
Retrieved: 2026-10-13.

Files: `src/probability.py` and `src/constants.py`. Only changes in the
vendored files are the relative import, `np.bool8` -> `np.bool_` for
NumPy 2 compatibility, and two `assumed =` -> `assumed |=` fixes in
`_assume_flags` and `_assume_safe_cells`. The original loops overwrote the
"any deduction" indicator when processing a later clue; this could count
invalid assignments before propagation reached a fixed point. A regression
fixture and 500 random small-board exhaustive-oracle comparisons verify the fix.
The project's Player and MineSweeper are NOT used.
No upstream NOTICE file was present in the inspected commit.

Integration and bounded execution are in `lib/csp_agent.py`. That adapter
subclasses the calculator to bound enumeration, suppress progress prints,
detect no-solution results and handle the empty-frontier case. It uses
the project's original environment, including its safe-but-not-necessarily-zero
first click. Upstream's center-first / zero-first board generator is never called.
