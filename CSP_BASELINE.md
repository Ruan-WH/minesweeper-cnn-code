# Independent CSP baseline

This implementation adds an observation-only traditional baseline without editing
the environment, constants, CNN/SP/SCL trainers, or existing CNN test scripts.
It is an adapted open-source CSP baseline, not a reproduction of upstream's
reported win rates or an optimal game-winning policy.

Completed final-version 1000-game results for all four mine counts, raw-log links,
validation checks and 10000-game runtime estimates are in [CSP_RESULTS.md](CSP_RESULTS.md).

## Inspected project protocol

- `minesweeper_pygame.py`: raw observation is a 2-D visible board; unknown = -1,
  revealed clues = 0..8, revealed mine = -2, flag = 9. `reset()` returns the board;
  `step(index)` returns `(board, reward, done, info)`. Index = row * columns + column.
- All agents here use left clicks, only on unknown cells, and stop immediately
  at `done`. Loss means a revealed mine; win means exactly `num_mines` cells
  remain unknown/flagged, i.e. every safe cell has been revealed. Flags need not
  be sent to the environment.
- The environment regenerates the WHOLE field if the first clicked cell is a
  mine, repeating until safe. It does not require a zero clue or an empty
  neighborhood. This original code is used unchanged.
- Existing `minesweeper_test_cnn_statistics.py` uses first index 0, one-hot input,
  `eval()`, and minimum model output among unknown cells. It runs 10 groups of
  1000 episodes (the comment saying 1000 groups is stale). For equal group sizes,
  mean group win rate equals total wins / total episodes. Group standard deviation
  is NOT a 95% confidence interval.
- Existing training prints a moving 500-game win rate with changing weights;
  it is not frozen-model evaluation, and its first click is model-selected.
- Training uses explicit 9x9 / 10 mines. Existing offline tests use constants;
  the inspected `constants.py` had 9x9 / 18 mines. The new harness uses explicit
  9x9 and `--mines`, avoiding this mismatch.
- No standalone Single-Point agent or SCL training program was found among the
  project's Python sources. Their unpublished/server versions were not inspected.
  The cheap SP-like reductions here are only a preprocessing step before actual
  constraint enumeration and globally weighted probability inference.

## Open-source selection and provenance

| Project / inspected commit | License | Dependencies / API | Decision |
| --- | --- | --- | --- |
| [nabenabe0928/mine-sweeper-solver](https://github.com/nabenabe0928/mine-sweeper-solver/tree/3a499ddee44bf7562808c8ab787bd173fb68fa2e) | Apache-2.0 | NumPy; `ProbabilityCalculator(cell_state, flags, neighbors, n_mines).compute()` | Selected: smallest direct probability interface, DFS with pruning, global combinatorial weighting. Its own board generator is excluded. |
| [JohnnyDeuss/minesweeper-solver](https://github.com/JohnnyDeuss/minesweeper-solver/tree/829a14fc07153c856d7200def14686c409953b5e) | No explicit license found in inspected repository/setup; do not assume copying permission | scipy==1.1.0, numpy==1.15.4, python-constraint==1.4.0; `Solver.solve(state)` | Strong component/area CLP approach, but old pins and uncertain copying permission complicate integration. No code copied. |
| [jgesc/Minesweeper_CSP](https://github.com/jgesc/Minesweeper_CSP/tree/73f1e13a31038b1a491683872b7e8348c4871200) | MIT | Python standard library; BaseAdapter, `MinesweeperCSP.step()` | Genuine constraint enumeration, but original guess uses `min(mine_counter)` (coordinate minimum); global total-mine combinatorial weighting is absent. More changes needed. No code copied. |

Upstream files are in `vendor/nabenabe_csp/`, with original LICENSE and
`UPSTREAM.md`. Modifications: relative import, NumPy boolean alias compatibility,
and two propagation-flag fixes. The latter was discovered by an exhaustive oracle:
later clues overwrote whether earlier clues produced new deductions, prematurely
accepting inconsistent configurations. The regression board is in the test suite.
The adapter additionally validates every counted leaf against visible clues.

## Agent and probability calculation

`CSPAgent.decide(visible_board)` has no environment parameter/reference. Only
a copied visible observation, declared rows/columns, and public mine count enter
the algorithm. It never imports the environment, its true board, or CNN labels.

For each unknown cell, define a Boolean variable (1 = mine). Each revealed clue
gives an equality on adjacent unknown variables. Deduced mines reduce its right
hand side. Single-clue and public-total-count deductions are propagated to a
fixed point; an available proven safe cell is clicked immediately. Otherwise
the vendored DFS enumerates valid assignments on the frontier, pruning clue
inconsistencies. Let an assignment use b frontier mines, f mines already proven,
U unconstrained cells and M public total mines. Its weight is
`choose(U, M - f - b)`. Incompatible global budgets contribute zero weight.
Each frontier cell's mine probability is its weighted mine count divided by
the total weight. Each unconstrained cell's numerator uses
`choose(U - 1, M - f - b - 1)`. Thus probabilities are exact (up to floating-point
arithmetic) conditional on visible clues under the uniform-placement prior,
when enumeration completes. The environment's whole-board rejection at a
fixed first click is consistent with this prior conditional on first-cell safety.

The agent minimizes immediate mine probability, not eventual game-loss probability.
Proven safe cells always take priority over positive-risk guesses, even if their
probabilities are within the numerical tie tolerance. Other ties within absolute
tolerance 1e-12 choose the lowest row-major index. Proven
mines are never deliberately opened. Each new observation is analyzed afresh;
no state is carried between episodes and no flag action is sent to the environment.

If no frontier constraint remains, the exact remaining-mine density is used and
recorded as `no_constraints` (also counted in `fallbacks` for transparency).
If there is no solution or the calculation budget expires, incomplete DFS counts
are discarded. The explicitly approximate fallback assigns the mean of adjacent
residual clue densities, or global remaining-mine density when no clue applies,
excludes already proven mines, and applies the same tie rule. These scores are
heuristics, not exact posteriors. No-solution propagation falls back without
retaining inconsistent inferred flags.

Default budget: 20000 DFS-loop iterations PER DECISION; no wall-clock cutoff.
This makes actions independent of machine speed and still bounds each search.
An optional `--timeout-seconds` also checks time every iteration, but timed
fallback trajectories can depend on hardware/load. A game takes at most 81 legal
clicks; unexpected errors are logged, included as non-wins, and cause a nonzero
batch exit. All losses and fallback games remain in the denominator.

## Running

Use the server's existing working environment, or install the small CSP dependency
set in a separate environment. Local validation used Python 3.12.13, NumPy 2.5.2,
gym 0.17.3 and pygame 2.6.1 on Windows. The AutoDL server is not connected to this
workspace, so its installed versions were not directly verified.

```bash
python -m pip install -r requirements-csp-baseline.txt
python -m unittest discover -s tests -p test_csp_baseline.py -v

# Smoke and stability checks; specify a NEW output directory for each run.
python minesweeper_test_csp_statistics.py --mines 10 14 18 22 --episodes 100 --seed 20261013 --output output/csp_baseline/server_smoke100
python minesweeper_test_csp_statistics.py --mines 10 14 18 22 --episodes 1000 --seed 20261013 --output output/csp_baseline/server_validation1000

# Paper-sized separate runs.
python minesweeper_test_csp_statistics.py --mines 10 --episodes 10000 --seed 20261013 --output output/csp_baseline/final10
python minesweeper_test_csp_statistics.py --mines 14 --episodes 10000 --seed 20261013 --output output/csp_baseline/final14
python minesweeper_test_csp_statistics.py --mines 18 --episodes 10000 --seed 20261013 --output output/csp_baseline/final18
python minesweeper_test_csp_statistics.py --mines 22 --episodes 10000 --seed 20261013 --output output/csp_baseline/final22
```

The CSP script has no torch dependency. To test the optional CNN adapter, use an
environment with PyTorch supporting `torch.load(..., weights_only=True)`:

The shared CNN adapter enables deterministic PyTorch algorithms and disables
cuDNN benchmarking. An unsupported nondeterministic CUDA operation is recorded
as an error rather than silently changing the protocol; CPU is also supported.

```bash
python -m unittest discover -s tests -p 'test_*baseline*.py' -v

# Use YOUR 10-mine checkpoint, and the same seed/count as final10.
python baseline_evaluation.py --agent cnn --mines 10 --episodes 10000 --seed 20261013 --model-path models/YOUR_10_MINE_MODEL.dat --cnn-dropout removed --device cuda --verify-boards output/csp_baseline/final10/episodes.csv --output output/csp_baseline/cnn10
```

Use `--cnn-dropout removed` if Dropout layers were commented out during training;
use `retained` if those layer slots exist (including Dropout with p=0). Checkpoint
loading is strict; the architecture must match. Run each other mine count with
its matching checkpoint and CSP reference file. No existing CNN script is edited.

## Fairness and reproducibility

Episode e uses NumPy seed `(base_seed + e) % 2**32` immediately before reset.
Both methods perform the same original reset and first click before any policy
decision. The environment uses no RNG after the first click, and the CSP policy
uses no RNG. Thus this harness's separate CNN and CSP runs get identical boards.
Merely seeding an entire arbitrary old script once is not a general pairing
guarantee: different RNG consumers or different first clicks can change later
boards and regeneration counts. Per-episode seeding avoids that problem without
changing the generator or distributing hidden boards to the agent.

The harness (not the agent) reads the hidden board ONLY to produce SHA256 audit
hashes before and after first-click regeneration. Neither the hashes nor the
board enter action selection. `--verify-boards` compares each episode's seeds
and both hashes against a reference CSV and aborts on mismatch. Different policies
were tested to produce the same first-click board hashes; repeated CSP runs were
tested for identical actions/outcomes. No pre-generated-board scheme is needed.

Each run writes:

- `metadata.json`: seed mapping, dimensions, mine counts, budgets, versions,
  source/model hashes, upstream commit and protocol.
- `episodes.csv`: every game, status, steps, latency, action/board hashes,
  fallback reasons/counts, enumeration iterations and errors.
- `summary.json`: wins, losses, errors, total episodes, overall win rate,
  two-sided Wilson 95% interval, elapsed seconds and budget/timeout counts.

The pilot and stability runs share their first 100 seeds. They are nested samples,
not independent replication. Do not combine their counts. The final runs above
also share the pilot prefix; use final-run results alone in the manuscript.

## Methods draft

We added an independent constraint-satisfaction and probabilistic baseline adapted
from the Apache-2.0 Python probability core of nabenabe0928/mine-sweeper-solver
(commit 3a499ddee44bf7562808c8ab787bd173fb68fa2e), with documented compatibility
and propagation fixes. The solver receives only revealed clues, board dimensions,
and the public total mine count. Unknown cells are Boolean variables constrained
by adjacent revealed clues. Following deterministic propagation, feasible frontier
assignments are enumerated and weighted by the number of compatible placements
in the unconstrained region. When no safe move is established, the solver chooses
the unknown cell with the lowest mine probability, breaking ties by row-major
index. Enumeration is limited to 20,000 iterations per decision; incomplete
enumerations are discarded and replaced by an explicitly logged clue-density
heuristic. Both methods are evaluated on 9 x 9 boards with 10, 14, 18, or 22 mines
using a fixed top-left initial click and the unchanged environment's first-click
safety rule, which does not guarantee a zero clue. Per-episode seeds and board
hashes enable identical-board comparisons. Win rates are reported with Wilson
95% confidence intervals, elapsed time, and all fallback counts; no unsuccessful
or budget-limited games are excluded.

For a final manuscript, explicitly add the actual number of games per condition,
seed, completed CNN pairing verification, and corresponding runtime hardware.
The draft describes the implemented protocol; it does not claim the unperformed
10,000-game runs or CNN comparisons have already been completed.
