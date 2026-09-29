"""Paired local screening. Cash margins are not Kaggle ladder ratings."""
import argparse
import json
from pathlib import Path
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

ROOT = Path(__file__).parent

def source(name):
    notebook = json.loads((ROOT / name).read_text())
    scope = {}
    exec(''.join(notebook['cells'][1]['source']), scope)
    return scope['SOURCE'].decode()

def run(seeds, output):
    herd = source('kaggriculture_adaptive_herd_safe_forecast_v3.ipynb')
    c95 = source('kaggriculture_adaptive_market_rhythm_c95.ipynb')
    archived = ROOT.parent / '_archive/kaggriculture-previous-notebooks-2026-09-26/kaggriculture_adaptive_farm_intelligence_v38_control.ipynb'
    v38 = next(''.join(c['source']).split('\n', 1)[1] for c in json.loads(archived.read_text())['cells'] if ''.join(c['source']).startswith('%%writefile main.py'))
    assert herd.count('_HP_WINDOW = 4') == 1
    variants = {'baseline': herd, 'window2': herd.replace('_HP_WINDOW = 4', '_HP_WINDOW = 2')}
    rows = []
    for seed in seeds:
        for opponent_name, opponent in {'c95': c95, 'v38': v38}.items():
            for seat in (0, 1):
                for name, raw in variants.items():
                    agents = [get_last_callable(opponent), get_last_callable(opponent)]
                    agents[seat] = get_last_callable(raw)
                    env = make('kaggriculture', configuration={'seed': seed, 'episodeSteps': 720})
                    env.run(agents)
                    final = env.steps[-1]
                    assert all(s.status == 'DONE' for s in final)
                    margin = final[seat].reward - final[1-seat].reward
                    row = dict(seed=seed, opponent=opponent_name, seat=seat, variant=name,
                               cash=final[seat].reward, margin=margin)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    (ROOT / output).write_text(json.dumps(rows, indent=2) + '\n')
    for name in variants:
        selected = [r for r in rows if r['variant'] == name]
        print(name, 'wins', sum(r['margin'] > 0 for r in selected), '/', len(selected),
              'mean margin', sum(r['margin'] for r in selected) / len(selected))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[280901, 280902])
    parser.add_argument('--output', default='paired_screening.json')
    args = parser.parse_args()
    run(args.seeds, args.output)
