"""Run a fixed CSP condition from the archived source snapshot."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
mines = int(sys.argv[1])
if mines not in (10, 14, 18, 22):
    raise ValueError('Unknown formal condition')
manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
for relative, expected in manifest['source_sha256'].items():
    actual = hashlib.sha256((HERE/'source_snapshot'/relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Frozen source changed: {relative}')
sys.path.insert(0, str(HERE/'source_snapshot'))
with (HERE/f'{mines}_run.log').open('x', encoding='utf-8') as log:
    with redirect_stdout(log), redirect_stderr(log):
        from baseline_evaluation import main
        main(['--agent', 'csp', '--mines', str(mines), '--episodes', '10000',
              '--seed', '20261013', '--max-iterations', '20000',
              '--timeout-seconds', '0', '--output', str(HERE/str(mines))])
