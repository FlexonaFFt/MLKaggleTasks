'''Check original-rank votes and exact preservation of the winning notebook.'''
import math
from build_consensus_backfill import build, HELPER
from build_aggressive_fusion_experiments import build as parent, METRIC_HELPER
from build_flow_experiments import validate

nb = build()
validate(nb)
winner = parent()[0]
assert nb['cells'][1:len(winner['cells'])] == winner['cells'][1:]
assert nb['metadata']['kaggle'] == winner['metadata']['kaggle']
env = {}
exec(METRIC_HELPER + HELPER, env)
identity = lambda s: None if s == 'invalid' else {'B-alt': 'B'}.get(s, s)
fuse = env['metric_fusion']
backfill = env['consensus_backfill']
first, second = ['A'], ['B', 'A']
base, scores, _ = fuse(first, second, identity)
candidates, revised, n = backfill(first, second, ['X'] * 25 + ['B', 'B-alt', 'A', 'invalid'], identity)
assert candidates == base and n == 1
assert math.isclose(revised[candidates.index('B')], scores[base.index('B')] + 1 / 29)
assert revised[candidates.index('A')] == scores[base.index('A')]
for tail in ([], ['B'] * 25, ['X'] * 26, ['A'] * 26):
    candidates, revised, n = backfill(first, second, tail, identity)
    assert (candidates, revised) == (base, scores) and n == 0
assert 'restored_tail_votes' in ''.join(nb['cells'][-1]['source'])
assert 'if not _collapsed:' in ''.join(nb['cells'][-1]['source'])
print('PASS: parent cell parity, exact tail ranks, single vote per identity, no-vote preservation, syntax')
