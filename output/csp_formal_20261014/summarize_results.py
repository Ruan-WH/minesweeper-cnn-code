"""Verify and summarize four completed, frozen 10,000-game conditions."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
results, group_records = [], []
for mines in (10, 14, 18, 22):
    folder = HERE/str(mines)
    with (folder/'episodes.csv').open(encoding='utf-8', newline='') as stream:
        episodes = list(csv.DictReader(stream))
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))[0]
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert len(episodes) == summary['episodes'] == 10000
    assert metadata['seed'] == 20261013
    assert metadata['max_iterations_per_decision'] == 20000
    assert metadata['timeout_seconds_per_decision'] == 0
    for relative, expected in metadata['source_sha256'].items():
        assert expected == manifest['source_sha256'][relative], relative
    for index, row in enumerate(episodes):
        assert int(row['episode']) == index
        assert int(row['seed']) == (20261013 + index) % 2**32
        assert int(row['mines']) == mines
        assert row['status'] in ('win', 'lose', 'error')
        assert int(row['max_search_iterations']) <= 20000
        assert row['initial_board_sha256'] and row['first_click_board_sha256']
    groups = []
    for group in range(10):
        block = episodes[group*1000:(group+1)*1000]
        wins = sum(row['status'] == 'win' for row in block)
        errors = sum(row['status'] == 'error' for row in block)
        groups.append(wins)
        group_records.append({'mines': mines, 'group': group+1, 'games': 1000,
                              'wins': wins, 'win_rate_percent': wins/10, 'errors': errors})
    assert sum(groups) == summary['wins']
    assert sum(row['status'] == 'lose' for row in episodes) == summary['losses']
    assert sum(row['status'] == 'error' for row in episodes) == summary['errors']
    for field, counter in [('iteration_limit','iteration_limit'),('timeouts','timeout'),
                           ('no_solution','no_solution'),('enumerations','enumerations'),
                           ('heuristic_fallbacks','heuristic_fallbacks'),('no_constraints','no_constraints')]:
        assert sum(int(row[field]) for row in episodes) == summary['counts'].get(counter, 0)
    decisions = sum(int(row['decision_count']) for row in episodes)
    assert decisions == summary['decision_count'] == summary['counts']['decisions']
    summary.update({'group_wins': groups, 'group_win_rates_percent': [w/10 for w in groups],
                    'group_SD_pp': statistics.pstdev([w/10 for w in groups]),
                    'win_rate_percent': summary['win_rate']*100,
                    'wilson_ci95_percent': [p*100 for p in summary['ci95']],
                    'unique_initial_board_hashes': len({r['initial_board_sha256'] for r in episodes}),
                    'base_seed': 20261013, 'max_iterations_per_decision': 20000,
                    'timeout_seconds': 0,
                    'raw_episode_csv_sha256': hashlib.sha256((folder/'episodes.csv').read_bytes()).hexdigest()})
    results.append(summary)

with (HERE/'group_wins.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, list(group_records[0]))
    writer.writeheader(); writer.writerows(group_records)
with (HERE/'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    fields = ['mines','episodes','wins','losses','errors','win_rate_percent','ci95_low_percent',
              'ci95_high_percent','group_SD_pp','mean_game_seconds','mean_decision_seconds',
              'max_search_iterations','decision_count','deduced_safe_decisions','enumerations',
              'iteration_limit_decisions','timeout_decisions','no_solution_decisions',
              'no_constraints_decisions','heuristic_fallback_decisions',
              'enumeration_success_decision_rate','iteration_limit_decision_rate',
              'heuristic_fallback_decision_rate','games_with_budget_exhaustion',
              'games_with_timeout','games_with_no_solution','games_with_no_constraints',
              'games_with_heuristic_fallback','games_with_solver_issue','seconds']
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for row in results:
        record = {field: row.get(field) for field in fields}
        record.update(ci95_low_percent=row['wilson_ci95_percent'][0],
                      ci95_high_percent=row['wilson_ci95_percent'][1],
                      deduced_safe_decisions=row['counts'].get('deduced_safe',0),
                      enumerations=row['counts'].get('enumerations',0))
        for field, counter in [('iteration_limit_decisions','iteration_limit'),
                               ('timeout_decisions','timeout'),('no_solution_decisions','no_solution'),
                               ('no_constraints_decisions','no_constraints'),
                               ('heuristic_fallback_decisions','heuristic_fallbacks')]:
            record[field] = row['counts'].get(counter,0)
        writer.writerow(record)
verification = {'completed_games':40000,'all_seed_mappings_verified':True,
                'all_recorded_source_hashes_match_frozen_manifest':True,
                'all_group_and_solver_counts_reconciled':True,
                'total_errors':sum(r['errors'] for r in results),
                'group_sd_definition':'population SD (ddof=0), percentage points',
                'timing_note':'Concurrent CPU-only runs; game timer excludes reset and initial board hash; decision timer excludes forced first click.',
                'comparison_note':'Unified game rules, independent random evaluation. Not paired against the previous CNN statistics runs.',
                'results':results}
(HERE/'all_results.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
manifest['status']='completed' if verification['total_errors']==0 else 'completed_with_errors'
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
