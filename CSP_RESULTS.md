# CSP baseline validation results

Final implementation, 2026-10-13. Each condition has 1000 games; these are stability results, not completed 10000-game publication runs.

9x9; first click = 0; original safe-first-click rule; base seed = 20261013; per-episode seeds = 20261013 through 20262012; 20000 iterations/decision; wall-clock timeout disabled.

| Mines | Wins / games | Win rate | Wilson 95% CI | Wall seconds | Fallback decisions | Budget-limited decisions / games | Timeouts | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 902 / 1000 | 90.2% | 88.20% - 91.89% | 41.64 | 1 | 0 / 0 | 0 | 0 |
| 14 | 746 / 1000 | 74.6% | 71.81% - 77.20% | 173.35 | 4 | 0 / 0 | 0 | 0 |
| 18 | 461 / 1000 | 46.1% | 43.03% - 49.20% | 464.94 | 14 | 0 / 0 | 0 | 0 |
| 22 | 215 / 1000 | 21.5% | 19.07% - 24.15% | 769.55 | 31 | 6 / 5 | 0 | 0 |

Fallback counts include exact uniform-density decisions when no frontier constraints remain. The corresponding no-constraint decision counts are 1, 4, 14, and 25. There were no no-solution events. For 22 mines, 6 searches in 5 games exhausted the iteration budget; partial enumeration counts were discarded and the documented heuristic was used. All 1000 games remain in that condition's denominator. No wall-clock timeouts are reported because that cutoff was disabled.

Elapsed times are actual per-process wall times and include the environment and CSV logging. The four conditions were launched concurrently; these times are exploratory throughput measurements, not isolated solver-vs-CNN latency benchmarks. Hardware reports Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, with 32 logical CPUs, Windows 11. Python 3.12.13, NumPy 2.5.2, gym 0.17.3, pygame 2.6.1. CSP uses CPU only.

## Raw artifacts

- [10-mine metadata](output/csp_baseline/final_validation1000/10/metadata.json), [episodes](output/csp_baseline/final_validation1000/10/episodes.csv), [summary](output/csp_baseline/final_validation1000/10/summary.json).
- [14-mine metadata](output/csp_baseline/final_validation1000/14/metadata.json), [episodes](output/csp_baseline/final_validation1000/14/episodes.csv), [summary](output/csp_baseline/final_validation1000/14/summary.json).
- [18-mine metadata](output/csp_baseline/final_validation1000/18/metadata.json), [episodes](output/csp_baseline/final_validation1000/18/episodes.csv), [summary](output/csp_baseline/final_validation1000/18/summary.json).
- [22-mine metadata](output/csp_baseline/final_validation1000/22/metadata.json), [episodes](output/csp_baseline/final_validation1000/22/episodes.csv), [summary](output/csp_baseline/final_validation1000/22/summary.json).

## Checks completed

- 10 unit/integration tests passed, including 500 random small-board exhaustive-oracle comparisons, a regression for the upstream propagation bug, inference beyond single-point deductions, valid-action checks, no-solution, budget, timeout, original first-click safety, terminal win/loss, deterministic episode replay, and CNN adapter compatibility with retained/removed Dropout slots.
- A 100-game 10-mine pilot was repeated with identical seeds, boards, action hashes, results and solver counts. The final 1000-game 10-mine run also exactly matches the previous run in those fields. A different policy produced the same first-click board hashes in integration tests.
- Final-version forced iteration-limit and forced-timeout batches each completed 10 games without runtime errors, logged 41 fallback decisions, and included all 10 losses in the denominator. These are stress checks, not primary benchmark results.
- All 4000 final episodes were checked for row counts, seed mapping, wins/losses, solver counters and recorded source SHA256 hashes.
- No original environment, constant, model, CNN training or existing CNN test file was edited during CSP integration.

## 10000-game planning

| Mines | Rough wall minutes for 10000 games |
| --- | --- |
| 10 | 6.9 |
| 14 | 28.9 |
| 18 | 77.5 |
| 22 | 128.3 |

These are linear extrapolations from the observed 1000-game wall times, not guarantees: CSP state difficulty, budget hits, CPU load and server hardware affect runtime. Run on AutoDL with the provided commands and record the actual final-run metadata. Pilot/stability samples share seed prefixes and must not be combined as independent samples.

Actual CNN checkpoint comparisons were not run because matching checkpoints for all four mine counts were not identified. The shared harness supports those paired tests and verifies initial/post-first-click board hashes; synthetic-checkpoint adapter tests are not research performance results. No standalone SP source was present in the inspected project.

Selection, licensing, algorithm, commands, and an English Methods draft: [CSP_BASELINE.md](CSP_BASELINE.md).
