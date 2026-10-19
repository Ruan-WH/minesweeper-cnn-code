"""Independent arithmetic checks and deterministic replay; never edits formal data."""
import csv
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
CHECK = HERE/'reliability_check'
CHECK.mkdir(exist_ok=True)
FROZEN = HERE/'source_snapshot'


def numerical_checks():
    manifest = json.loads((HERE/'manifest.json').read_text())
    combined = json.loads((HERE/'all_results.json').read_text())
    reported = {int(r['mines']):r for r in combined['results']}
    with (HERE/'results.csv').open(encoding='utf-8-sig') as f:
        table = {int(r['mines']):r for r in csv.DictReader(f)}
    with (HERE/'group_wins.csv').open(encoding='utf-8-sig') as f:
        group_table = {(int(r['mines']),int(r['group'])):r for r in csv.DictReader(f)}
    workspace = HERE.parents[1]
    current_matches = {}
    for relative, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((FROZEN/relative).read_bytes()).hexdigest() == expected
        p = workspace/relative
        current_matches[relative] = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == expected
    results = []
    for mines in (10,14,18,22):
        folder = HERE/str(mines)
        with (folder/'episodes.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows)==10000
        summary = json.loads((folder/'summary.json').read_text())[0]
        assert summary == {k:reported[mines][k] for k in summary}
        metadata = json.loads((folder/'metadata.json').read_text())
        assert (metadata['rows'],metadata['cols'],metadata['mines'])==(9,9,[mines])
        assert metadata['seed']==20261013 and metadata['max_iterations_per_decision']==20000
        assert metadata['timeout_seconds_per_decision']==0 and metadata['first_click']==0
        for file,digest in metadata['source_sha256'].items():
            assert digest==manifest['source_sha256'][file]
        for i,r in enumerate(rows):
            assert int(r['episode'])==i and int(r['seed'])==20261013+i
            assert int(r['mines'])==mines and r['status'] in ('win','lose')
            assert not r['error'] and json.loads(r['no_solution_diagnostics'])==[]
            assert 1<=int(r['steps'])<=81
            assert int(r['decision_count'])==int(r['steps'])-1
            reasons=json.loads(r['decision_reasons'])
            assert sum(reasons.values())==int(r['decision_count'])
            assert reasons.get('enumerated',0)==int(r['enumerations'])
            assert int(r['exact_decisions'])+int(r['heuristic_fallbacks'])==int(r['decision_count'])
            assert int(r['fallbacks'])==int(r['no_constraints'])+int(r['heuristic_fallbacks'])
            assert int(r['heuristic_fallbacks'])==sum(int(r[k]) for k in ('iteration_limit','timeouts','no_solution'))
            assert 0<=int(r['max_search_iterations'])<=20000
            for key in ('initial_board_sha256','first_click_board_sha256','actions_sha256'):
                assert len(r[key])==64 and all(c in '0123456789abcdef' for c in r[key])
        assert len({r['seed'] for r in rows})==10000
        assert len({r['initial_board_sha256'] for r in rows})==10000
        assert len({r['first_click_board_sha256'] for r in rows})==10000
        wins=sum(r['status']=='win' for r in rows)
        groups=[sum(r['status']=='win' for r in rows[i:i+1000]) for i in range(0,10000,1000)]
        rates=[g/10 for g in groups]
        sd=math.sqrt(math.fsum((p-wins/100)**2 for p in rates)/10)
        with localcontext() as ctx:
            ctx.prec=50
            n=Decimal(10000); w=Decimal(wins)
            z=Decimal(str(statistics.NormalDist().inv_cdf(0.975)))
            center=(w+z*z/2)/(n+z*z)
            radius=z/(n+z*z)*(w*(n-w)/n+z*z/4).sqrt()
            ci=[float(center-radius),float(center+radius)]
        assert wins==summary['wins']==int(table[mines]['wins'])
        assert abs(summary['win_rate']-wins/10000)<1e-14
        assert all(abs(a-b)<1e-12 for a,b in zip(ci,summary['ci95']))
        assert abs(sd-reported[mines]['group_SD_pp'])<1e-12
        assert groups==reported[mines]['group_wins']
        assert abs(float(table[mines]['group_SD_pp'])-sd)<1e-12
        for key in ('ci95_low_percent','ci95_high_percent'):
            assert abs(float(table[mines][key])-100*ci['high' in key])<1e-10
        for i,g in enumerate(groups,1):
            assert int(group_table[mines,i]['wins'])==g
            assert int(group_table[mines,i]['games'])==1000
            assert float(group_table[mines,i]['win_rate_percent'])==g/10
        count_map={'iteration_limit':'iteration_limit','timeouts':'timeout','no_solution':'no_solution',
                   'no_constraints':'no_constraints','heuristic_fallbacks':'heuristic_fallbacks',
                   'fallbacks':'fallbacks','enumerations':'enumerations','enumeration_iterations':'enumeration_iterations'}
        for col,key in count_map.items():
            assert sum(int(r[col]) for r in rows)==summary['counts'].get(key,0)
        decisions=sum(int(r['decision_count']) for r in rows)
        assert decisions==summary['decision_count']==summary['counts']['decisions']
        assert sum(json.loads(r['decision_reasons']).get('deduced_safe',0) for r in rows)==summary['counts']['deduced_safe']
        games_map={'games_with_budget_exhaustion':'iteration_limit','games_with_timeout':'timeouts',
                   'games_with_no_solution':'no_solution','games_with_no_constraints':'no_constraints',
                   'games_with_heuristic_fallback':'heuristic_fallbacks','games_with_fallback':'fallbacks'}
        for key,col in games_map.items():
            assert sum(int(r[col])>0 for r in rows)==summary[key]
        assert sum(any(int(r[k])>0 for k in ('iteration_limit','timeouts','no_solution')) for r in rows)==summary['games_with_solver_issue']
        assert abs(math.fsum(float(r['seconds']) for r in rows)/10000-summary['mean_game_seconds'])<1e-10
        assert abs(math.fsum(float(r['decision_seconds']) for r in rows)/decisions-summary['mean_decision_seconds'])<1e-10
        assert max(int(r['max_search_iterations']) for r in rows)==summary['max_search_iterations']
        assert hashlib.sha256((folder/'episodes.csv').read_bytes()).hexdigest()==reported[mines]['raw_episode_csv_sha256']
        log=(HERE/f'{mines}_run.log').read_text()
        assert 'Traceback' not in log and 'Results:' in log
        results.append({'mines':mines,'wins':wins,'groups':groups,'group_SD_pp':sd,'ci95':ci,
                        'unique_initial_boards':10000,'unique_first_click_boards':10000})
    report={'passed':True,'checked_games':40000,'method':'Raw-row reconciliation; independently implemented 50-digit Decimal Wilson calculation and fsum-based population SD.',
            'all_frozen_sources_verified':True,'workspace_matches_frozen_sources':current_matches,'results':results}
    (CHECK/'independent_arithmetic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def replay(mines):
    sys.path.insert(0,str(FROZEN))
    import numpy as np
    import baseline_evaluation as evaluation
    from lib.csp_agent import CSPAgent
    from minesweeper_pygame import MinesweeperEnv
    for module in (evaluation,sys.modules[CSPAgent.__module__],sys.modules[MinesweeperEnv.__module__]):
        assert Path(module.__file__).resolve().is_relative_to(FROZEN.resolve())
    with (HERE/str(mines)/'episodes.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    selected={i for i in range(10000) if i%1000 in (0,999)}
    selected.update(random.Random(20261014+mines).sample(range(10000),10))
    affected=[i for i,r in enumerate(rows) if int(r['heuristic_fallbacks'])>0]
    selected.update(affected)
    selected.update([i for i,r in enumerate(rows) if int(r['no_constraints'])>0][:3])
    selected.update([i for i,r in enumerate(rows) if r['initial_board_sha256']!=r['first_click_board_sha256']][:3])
    records=[]; decisions_checked=0
    for index in sorted(selected):
        env=MinesweeperEnv(9,9,mines,'rgb_array')
        policy=CSPAgent(9,9,mines,max_iterations=20000,timeout_seconds=0)
        original=policy.decide
        def checked_decide(visible):
            nonlocal decisions_checked
            decision=original(visible)
            assert np.array_equal(visible,env.player_board)
            assert np.all((visible>=-1)&(visible<=8))
            assert visible.reshape(-1)[decision.action]==-1
            assert decision.action not in decision.inferred_mines
            # Test-oracle access AFTER selection only; never fed to the policy.
            truth=env.board.reshape(-1)
            for cell in decision.inferred_mines:
                assert truth[cell]==-2
            if decision.exact:
                unknown=np.flatnonzero(visible.reshape(-1)==-1)
                safe=[j for j in unknown if decision.probabilities[j]==0]
                assert all(truth[j]!=-2 for j in safe)
            decisions_checked+=1
            return decision
        policy.decide=checked_decide
        try:
            actual=evaluation.play_episode(env,policy,int(rows[index]['seed']),index,mines)
            assert not actual['error'],actual['error']
            for key,value in actual.items():
                if key not in ('seconds','decision_seconds','mean_decision_seconds'):
                    assert str(value)==rows[index][key],(mines,index,key,value,rows[index][key])
            records.append({'episode':index,'seed':actual['seed'],'status':actual['status'],
                            'steps':actual['steps'],'iteration_limit':actual['iteration_limit'],
                            'actions_sha256':actual['actions_sha256']})
        finally:
            env.close()
    report={'passed':True,'mines':mines,'replayed_games':len(records),'decisions_checked':decisions_checked,
            'all_heuristic_affected_games_replayed':len(affected),
            'non_timing_fields_match':True,'test_oracle_note':'Truth accessed only after action selection to validate deductions; it does not influence the replayed policy.',
            'records':records}
    (CHECK/f'replay_{mines}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':
    if len(sys.argv)==1:
        numerical_checks()
    else:
        replay(int(sys.argv[1]))
