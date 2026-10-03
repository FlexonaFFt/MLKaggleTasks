"""Build two self-contained CASMI26 experiments from the confirmed 0.409 notebook."""

import ast
import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE / "casmi26_exp_409_libgate_pop025.ipynb"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old[:70]!r}; got {text.count(old)}")
    return text.replace(old, new, 1)


def source(cell):
    return "".join(cell["source"])


def set_source(cell, value):
    cell["source"] = value.splitlines(keepends=True)


def embedded_sources(notebook):
    tree = ast.parse(source(notebook["cells"][2]))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "EMBED" for target in node.targets))
    return ast.literal_eval(assignment.value)


def add_defaults(fusion, new_defaults):
    return replace_once(fusion, "FILL_25=False,                         # top up lists shorter than 25 from the other channels",
                        "FILL_25=False, " + new_defaults + ",  # experiment switches; default keeps 0.409 behavior")


PROMOTE_FUNCTION = '''    def promote_pc(mid, lst):
        """Promote the already-listed PubChem leader only when independent evidence agrees."""
        p = pc.get(mid) or pc.get(str(mid)) or {}
        if not c['PROMOTE_PC_CONSENSUS'] or len(lst) < 2 or lib_of.get(mid, 0.0) >= LIB_TAU:
            return lst
        if not p.get('pc_keys') or not p.get('pc_fz'):
            return lst
        bp = p.get('best_pool_fz')
        if bp is None or not np.isfinite(bp) or p['pc_fz'][0] - bp <= c['REL_TH']:
            return lst
        wanted = p['pc_keys'][0]
        rank = next((i for i, s in enumerate(lst[:5]) if i and score_key(s) == wanted), None)
        if rank is None or formula(lst[rank]) != formula(lst[0]):
            return lst
        ice = ice_scores.get(str(mid), {}) if isinstance(ice_scores, dict) else {}
        gl = gl_of(mid)
        def wins(scores):
            a, b = scores.get(lst[rank]), scores.get(lst[0])
            return a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and a > b
        # A measured GLACIER comparison must agree. If unavailable, require a larger FP gap.
        gl_available = gl.get(lst[rank]) is not None and gl.get(lst[0]) is not None
        if not wins(ice) or (gl_available and not wins(gl)) or (not gl_available and p['pc_fz'][0] - bp <= 2 * c['REL_TH']):
            return lst
        stats.setdefault('pc_promotion', {'molecules': 0})['molecules'] += 1
        return [lst[rank]] + lst[:rank] + lst[rank + 1:]

'''


TAIL_FUNCTION = '''    def diversify_tail(mid, lst):
        """Keep the first 20 ranks intact; spend at most five tail slots on unseen PubChem keys."""
        if not c['TAIL_PC_SLOTS'] or len(lst) <= 20 or lib_of.get(mid, 0.0) >= LIB_TAU:
            return lst
        p = pc.get(mid) or pc.get(str(mid)) or {}
        bp = p.get('best_pool_fz')
        if not p.get('pc') or not p.get('pc_fz') or bp is None or not np.isfinite(bp):
            return lst
        if p['pc_fz'][0] <= bp:
            return lst
        seen = {score_key(s) or s for s in lst}
        novel = []
        for s, k in zip(p['pc'], p['pc_keys']):
            k = k or score_key(s) or s
            if k not in seen:
                seen.add(k); novel.append(s)
        count = min(int(c['TAIL_PC_SLOTS']), len(lst) - 20, len(novel))
        if not count:
            return lst
        stats.setdefault('pc_tail', {'molecules': 0, 'inserted': 0})['molecules'] += 1
        stats['pc_tail']['inserted'] += count
        return lst[:-count] + novel[:count]

'''


def build(kind, filename, title, explanation):
    notebook = copy.deepcopy(json.loads(BASE.read_text()))
    opening = source(notebook['cells'][0])
    opening = replace_once(
        opening,
        "'''CASMI26 | Library-Gated Fusion + Popularity Prior\nFaithful adaptation of the public notebook below. Its reported public score 0.409 is not reproduced here.\n",
        f"'''CASMI26 | {title}\n{explanation}\nBaseline: our adaptation has confirmed public LB 0.409. This experiment has not been scored.\n",
    )
    set_source(notebook['cells'][0], opening)

    embedded = embedded_sources(notebook)
    fusion = embedded['fusion_core.py']
    if kind == 'consensus':
        fusion = add_defaults(fusion, 'PROMOTE_PC_CONSENSUS=False')
        fusion = replace_once(fusion, "    # ---- stage C: engine-2 fusion (+ ICE/GL on the fused list)",
                              PROMOTE_FUNCTION + "    # ---- stage C: engine-2 fusion (+ ICE/GL on the fused list)")
        fusion = replace_once(fusion, "        vs = fill(mid, vs[:25])\n        out.append((mid, ';'.join(vs[:25]) if vs else c['FALLBACK']))",
                              "        vs = promote_pc(mid, fill(mid, vs[:25]))\n        out.append((mid, ';'.join(vs[:25]) if vs else c['FALLBACK']))")
        fusion = replace_once(fusion, "v4 = [(mid, ';'.join(fill(mid, [x for x in s.split(';') if x and x != c['FALLBACK']])) or c['FALLBACK'])",
                              "v4 = [(mid, ';'.join(promote_pc(mid, fill(mid, [x for x in s.split(';') if x and x != c['FALLBACK']]))) or c['FALLBACK'])")
        config = "CFG.update({'VERSION': 'ours-409-pc-consensus-promotion', 'PROMOTE_PC_CONSENSUS': True})\n"
    else:
        fusion = add_defaults(fusion, 'TAIL_PC_SLOTS=0')
        fusion = replace_once(fusion, "    # ---- stage C: engine-2 fusion (+ ICE/GL on the fused list)",
                              TAIL_FUNCTION + "    # ---- stage C: engine-2 fusion (+ ICE/GL on the fused list)")
        fusion = replace_once(fusion, "        vs = fill(mid, vs[:25])\n        out.append((mid, ';'.join(vs[:25]) if vs else c['FALLBACK']))",
                              "        vs = diversify_tail(mid, fill(mid, vs[:25]))\n        out.append((mid, ';'.join(vs[:25]) if vs else c['FALLBACK']))")
        fusion = replace_once(fusion, "v4 = [(mid, ';'.join(fill(mid, [x for x in s.split(';') if x and x != c['FALLBACK']])) or c['FALLBACK'])",
                              "v4 = [(mid, ';'.join(diversify_tail(mid, fill(mid, [x for x in s.split(';') if x and x != c['FALLBACK']]))) or c['FALLBACK'])")
        config = "CFG.update({'VERSION': 'ours-409-pc-tail-coverage', 'FILL_25': True, 'TAIL_PC_SLOTS': 5})\n"
    embedded['fusion_core.py'] = fusion
    set_source(notebook['cells'][2], "# embedded sources (self-contained Kaggle notebook)\nEMBED = " + repr(embedded) + "\n")
    set_source(notebook['cells'][4], source(notebook['cells'][4]).rstrip() + "\n" + config)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    path = HERE / filename
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(path)


if __name__ == '__main__':
    build('consensus', 'casmi26_exp_409_pc_consensus_promotion.ipynb',
          'Selective PubChem Consensus Promotion',
          'Move the leading PubChem-only candidate to rank 1 only for low-library-confidence molecules '
          'when fingerprint and forward-spectrum evidence agree. No new datasets.')
    build('tail', 'casmi26_exp_409_pc_tail_coverage.ipynb',
          'PubChem Tail Coverage',
          'Keep ranks 1-20, backfill short lists, and reserve up to five tail slots for unseen PubChem structures '
          'on low-library-confidence molecules. No new datasets.')
