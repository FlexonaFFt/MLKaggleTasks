'''Synthetic coverage and isolation checks for the appended tail experiment.'''
import json

from build_flow_experiments import BASE, ROOT, source, validate
from build_tail_consensus import TAIL_HELPER


if __name__ == '__main__':
    namespace = dict(ALPHA=.6, KRR=3.)
    exec(TAIL_HELPER, namespace)
    tail = namespace['consensus_tail']
    key = lambda s: s.removesuffix('_alias') if s != 'INVALID' else None
    final = [f'b{i}' for i in range(25)]
    result, added = tail(final, ['x', 'y'] + final[:20], ['y', 'x'] + final[:20], key)
    assert added == ['x', 'y']
    assert result == final[:20] + ['x', 'y'] + final[20:23]
    assert final == [f'b{i}' for i in range(25)], 'Inputs mutated'
    assert tail(final, ['x'], ['z'], key) == (final, [])
    assert tail(final, [], ['x'], key) == (final, [])
    assert tail(final, final, final, key) == (final, [])
    assert tail(final[:20], ['x'], ['x'], key) == (final[:20], [])
    assert tail(['CCO'], ['x'], ['x'], key) == (['CCO'], [])
    assert tail(final, ['b0_alias', 'INVALID'], ['b0', 'INVALID'], key) == (final, [])
    assert tail(final, final + ['x'], final + ['x'], key) == (final, []), 'Only top-25 may qualify'
    assert tail(final[:21], ['x', 'y'], ['x', 'y'], key)[0] == final[:20] + ['x']
    extras = [f'x{i}' for i in range(10)]
    result, added = tail(final, extras, extras, key)
    assert result == final[:20] + extras[:5] and len(added) == 5
    result, added = tail(final, ['x_alias', 'x', 'y'], ['x', 'y'], key)
    assert len(added) == 2 and len({key(s) for s in result}) == 25
    try:
        tail(final[:24] + ['b0_alias'], ['x'], ['x'], key)
    except AssertionError:
        pass
    else:
        raise AssertionError('Canonical baseline duplicates must be rejected')
    base = json.loads(BASE.read_text())
    nb = json.loads((ROOT / 'experiments/casmi26_0420_top20_locked_consensus_tail.ipynb').read_text())
    validate(nb)
    assert nb['metadata']['kaggle'] == base['metadata']['kaggle']
    assert len(nb['cells']) == len(base['cells']) + 1
    assert source(nb, 27) == "TAIL_PRIMARY = pd.read_csv('submission.csv').copy()\n" + source(base, 27)
    assert all(source(nb, i) == source(base, i) for i in range(7, 27))
    print('PASS: tail ordering, top20 lock, no-ops, top25 scope, aliases, short lists, duplicates, upstream parity and syntax')
