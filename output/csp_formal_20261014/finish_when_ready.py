"""Automatically verify and summarize the formal runs after all conditions finish."""
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import runpy
import time
import traceback

HERE = Path(__file__).resolve().parent
started = time.monotonic()
with (HERE/'completion.log').open('x', encoding='utf-8') as log:
    with redirect_stdout(log), redirect_stderr(log):
        try:
            while True:
                ready = True
                for mines in (10, 14, 18, 22):
                    path = HERE/str(mines)/'summary.json'
                    try:
                        summary = json.loads(path.read_text(encoding='utf-8'))[0]
                        ready &= summary['episodes'] == 10000
                    except (FileNotFoundError, json.JSONDecodeError, IndexError):
                        ready = False
                    run_log = HERE/f'{mines}_run.log'
                    if run_log.exists() and 'Traceback (most recent call last)' in run_log.read_text(encoding='utf-8'):
                        raise RuntimeError(f'{mines}-mine process recorded an exception; raw games are retained')
                if ready:
                    break
                if time.monotonic()-started > 8*60*60:
                    raise RuntimeError('Completion monitor reached eight hours; no games were discarded or marked lost. Inspect running processes and partial logs.')
                time.sleep(15)
            runpy.run_path(str(HERE/'summarize_results.py'), run_name='__main__')
            report = json.loads((HERE/'all_results.json').read_text(encoding='utf-8'))
            rows = report['results']
            text = ['Formal CSP evaluation: 9 x 9; 10,000 games per mine count; 40,000 games total.',
                    'Base seed 20261013; per-game seed = (base + zero-based index) modulo 2^32.',
                    '20,000 iterations/decision; wall-clock cutoff disabled; frozen source snapshot.',
                    'Win-rate CIs are Wilson 95%; group SD is population SD across ten 1,000-game groups (pp).',
                    'These evaluation groups are not independent training runs.',
                    'Unified game rules with CNN, independent random testing; not board-by-board paired.',
                    'Concurrent local CPU timings are not isolated serial latency benchmarks.', '',
                    'Mines | Wins/10000 | Win rate | Wilson 95% CI | Group SD (pp)']
            for r in rows:
                lo,hi = r['wilson_ci95_percent']
                text.append(f"{r['mines']} | {r['wins']}/10000 | {r['win_rate_percent']:.2f}% | {lo:.2f}-{hi:.2f}% | {r['group_SD_pp']:.2f}")
            text.append('\nDiagnostics (decision counts exclude forced first click):')
            for r in rows:
                c=r['counts']
                text.append(f"{r['mines']} mines: decisions={r['decision_count']}, rule-safe={c.get('deduced_safe',0)}, enumerations={c.get('enumerations',0)}, iteration-limit={c.get('iteration_limit',0)}, timeout={c.get('timeout',0)}, no-solution={c.get('no_solution',0)}, no-frontier={c.get('no_constraints',0)}, heuristic-fallback={c.get('heuristic_fallbacks',0)}, affected games={r['games_with_solver_issue']}, errors={r['errors']}; mean/game={r['mean_game_seconds']:.6f}s, mean/decision={r['mean_decision_seconds']:.6f}s, max iterations={r['max_search_iterations']}")
                text.append('Group wins: '+', '.join(map(str,r['group_wins'])))
            (HERE/'formal_results.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
            r18 = next(r for r in rows if r['mines']==18)
            draft = ('3.5. Comparison with a Constraint-Based Solver\n\n'
                'We evaluated a constraint-based solver adapted from nabenabe0928/mine-sweeper-solver '
                '(commit 3a499ddee44bf7562808c8ab787bd173fb68fa2e; Apache-2.0). It uses only revealed clues, '
                'unknown-cell locations, and the public mine count. Deterministic propagation establishes '
                'safe cells and mines; otherwise, feasible frontier configurations are enumerated and '
                'weighted by the number of compatible non-frontier mine placements. The solver selects '
                'the lowest-risk unknown cell, with reproducible row-major tie breaking.\n\n'
                'Enumeration is limited to 20,000 iterations per decision, without a wall-clock cutoff. '
                'Incomplete counts are discarded if the budget is exceeded, and a logged heuristic risk '
                'estimate is used. Exact uniform probabilities in states without frontier constraints are '
                'recorded separately. We used 10,000 games per mine count on 9 x 9 boards, grouped into ten '
                'consecutive batches of 1,000 games. Episode seeds were (20261013 + episode index) modulo '
                '2^32. The first click was the top-left cell, with the original safe-but-not-necessarily-zero '
                'rule and terminal conditions. This is a unified evaluation protocol, not verified '
                'board-by-board pairing with the previously reported CNN runs.\n\n')
            rates='; '.join(f"{r['win_rate_percent']:.2f}% for {r['mines']} mines" for r in rows)
            draft += ('The constraint-based solver achieved '+rates+'. Wilson confidence intervals, '
                'evaluation-group standard deviations, and search-budget diagnostics are supplied with '
                'the formal results. At 18 mines, its measured win rate was '
                f"{r18['win_rate_percent']:.2f}%, compared with 20.38% for the evaluated SCL online-best "
                'checkpoint. The corresponding SP comparison must be inserted from documented SP '
                'measurements; no SP value is inferred here.\n\n'
                'This comparison distinguishes explicit constraint reasoning from staged CNN training. '
                'The contribution of SCL is the training strategy within the evaluated CNN family, rather '
                'than a general claim of superiority to traditional solvers. Evaluation-group variability '
                'does not establish robustness across independent training seeds. Best-checkpoint and '
                'fixed-budget endpoint comparisons should also be distinguished.\n')
            (HERE/'section_3_5_draft.txt').write_text(draft, encoding='utf-8')
            print('All formal results verified and written.', flush=True)
        except Exception:
            traceback.print_exc()
            (HERE/'completion_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise
