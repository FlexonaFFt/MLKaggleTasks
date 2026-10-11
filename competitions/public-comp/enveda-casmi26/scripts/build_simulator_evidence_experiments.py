'''Two independent same-run overlays on exact E006, using existing simulator evidence only.'''
import copy
import json
from build_ice_fallback_versions import SOURCE, build as e006_build
from build_metric_safe_adaptation import code_cell, comments
from build_flow_experiments import ROOT, prose, replace, validate

HELPERS = '''
def finite_score(value):
    import math
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def aligned_simulator_scores(evidence, candidates, keys, known, identity, formula, key_of):
    import math
    groups = {}
    for smile in sorted(set(known + candidates)):
        source = smile
        value = evidence.get(source)
        if value is None:
            source = key_of(smile)
            value = evidence.get(source)
        ident = identity(smile)
        if ident is not None and finite_score(value):
            groups.setdefault((ident, formula(smile)), {})[source] = float(value)
    result, filled = dict(evidence), 0
    for smile, key in zip(candidates, keys):
        direct = evidence.get(smile)
        if direct is None:
            direct = evidence.get(key)
        # Preserve every original finite score; fill only absent/invalid evidence.
        if finite_score(direct):
            continue
        ident = identity(smile)
        aliases = groups.get((ident, formula(smile)), {}) if ident is not None else {}
        if aliases:
            result[smile] = math.fsum(aliases.values()) / len(aliases)
            filled += 1
    return result, filled

def complete_group_ice(candidates, keys, formulas, ice, gl):
    groups = {}
    for i, formula in enumerate(formulas):
        groups.setdefault(formula, []).append(i)
    accepted, n_groups = {}, 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = []
        for i in indices:
            a, b = ice.get(candidates[i]), gl.get(candidates[i])
            if a is None:
                a = ice.get(keys[i])
            if b is None:
                b = gl.get(keys[i])
            if not finite_score(a) or not finite_score(b):
                break
            values.append(float(a))
        else:
            n_groups += 1
            accepted.update((candidates[i], v) for i, v in zip(indices, values))
    return accepted, n_groups
'''

ALIGN_INJECTION = '''
        audit['aligned_ice_scores'] = audit['aligned_gl_scores'] = 0
        if ice_fallback:
            known = (list(scope['BASE_TAIL_CANDIDATES'].get(mid, []))
                + list(engine['smiles']) + list(scope['PREFUSION_LISTS'][mid]))
            pc = next((v for k, v in scope['PC'].items() if str(k) == mid), {})
            known += list(pc.get('pc', []))
            ice, ni = aligned_simulator_scores(ice, candidates, keys, known,
                scope['metric_identity'], scope['_form2'], scope['chem'].score_key)
            gl, ng = aligned_simulator_scores(gl, candidates, keys, known,
                scope['metric_identity'], scope['_form2'], scope['chem'].score_key)
            audit.update(aligned_ice_scores=ni, aligned_gl_scores=ng)
'''

JOINT_INJECTION = '''
        joint_ice = {}
        audit['joint_formula_groups'] = 0
        if ice_fallback and ice and gl:
            base = next((v for k, v in scope['BASE'].items() if str(k) == mid), ([], [], 1.0))
            if base[2] < scope['LIB_TAU']:
                joint_ice, ng = complete_group_ice(candidates, keys,
                    [scope['_form2'](s) for s in candidates], ice, gl)
                audit['joint_formula_groups'] = ng
'''

COMPARE = '''
_control = pd.read_csv('submission_e006_control.csv', dtype=str, keep_default_na=False)
_treatment = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
assert list(_control.molecule_id) == list(_treatment.molecule_id)
_paired = pd.DataFrame(dict(molecule_id=_control.molecule_id,
    changed=_control.smiles != _treatment.smiles,
    changed_top1=_control.smiles.str.split(';').str[0] != _treatment.smiles.str.split(';').str[0]))
_paired.to_csv('paired_e006_audit.csv', index=False)
_report = dict(experiment=EXPERIMENT, changed_rows=int(_paired.changed.sum()),
    changed_top1=int(_paired.changed_top1.sum()), control_sha256=sha256('submission_e006_control.csv'),
    submission_sha256=sha256('submission.csv'), measured_score=None)
Path('paired_e006_audit.json').write_text(json.dumps(_report, indent=2))
print(_report)
if not _report['changed_rows']:
    print('NO-OP versus E006: do not spend a separate submission on identical predictions.')
'''


def treatment_code(kind):
    original = json.loads(SOURCE.read_text())
    code = ''.join(original['cells'][-1]['source'])
    injection = ALIGN_INJECTION if kind == 'metric_aligned_scores' else JOINT_INJECTION
    code = replace(code, '        gl = scope["gl_of"](mid)\n', '        gl = scope["gl_of"](mid)\n' + injection)
    if kind == 'complete_group_joint_ice_gl':
        code = replace(code, '                [{}, gl],', '                [joint_ice, gl],')
    code = replace(code, "'experiment': 'E006', 'window': 40, 'ice_fallback': True",
                   f"'experiment': '{kind}', 'window': 40, 'ice_fallback': True")
    code = replace(code, "'helper_source_sha256':", "'inherited_helper_source_sha256':")
    return comments(code)


def build():
    base = e006_build()[0]
    notebooks = []
    for kind, title, reasoning in (
        ('metric_aligned_scores', 'Metric-Aligned Simulator Evidence',
         'Fusion chooses one SMILES representative per official metric identity, but simulator evidence '
         'is looked up by SMILES/internal key. Fill missing scores from already-scored equivalent '
         'representatives with the SAME molecular formula. Preserve every direct finite score. '
         'Use the arithmetic mean of available alias scores, never the maximum. This is a coverage '
         'hypothesis, not proof that metric-equivalent representations have identical spectra. '
         'No aliases means unchanged predictions. Evidence alignment may change the fallback branch.'),
        ('complete_group_joint_ice_gl', 'Complete-Coverage ICEBERG + GLACIER',
         'When GLACIER is present, add ICEBERG only for weak-library molecules (lib_max < 0.9) '
         'and same-formula groups with at least two candidates and finite scores for EVERY member '
         'from BOTH simulators. Use existing weights 1.0/1.0; retain the original GLACIER treatment '
         'for all other groups and the original ICE-only fallback where GLACIER is absent. '
         'Do not align alias scores in this experiment. Earlier broad ICE inclusion hurt scores; '
         'complete coverage is an unvalidated attempt to constrain that risk.')):
        nb = copy.deepcopy(base)
        nb['cells'][0] = code_cell(prose('CASMI26: ' + title + '\n'
            'Independent experimental overlay on the exact E006 window=40 ICE-fallback adaptation. '
            'Source score 0.440 was reported by the user; our E006/E007 results are still pending. '
            + reasoning + '\nNo window=80, consensus backfill, cross-formula ranking or new model pass. '
            'Same 17 inputs, Python 3.12 and T4 GPU. No score gain is guaranteed.'))
        nb['cells'].append(code_cell(prose('Experiment method and attribution\n' + reasoning + '\n'
            'Our contribution is this independent evidence-use experiment. Direct E006 upload, '
            '@samanyu1808 fallback idea and all upstream authors/links remain credited above. '
            'The complete E006 run is saved byte-for-byte before treatment. Control reconstruction '
            'uses unmodified scores and must match E004; the final paired audit compares to E006. '
            'No hidden labels or network access. Synthetic tests do not establish real-world score coverage.')
            + HELPERS + f"\nEXPERIMENT = '{kind}'\n"
            + "import shutil\n"
            + "shutil.copy2('submission.csv', 'submission_e006_control.csv')\n"
            + "shutil.copy2('experiment_audit_factorial.csv', 'e006_audit.csv')\n"
            + "shutil.copy2('experiment_verification.json', 'e006_verification.json')\n"
            + "shutil.copy2('submission_e004_control.csv', 'submission.csv')\n"
            + treatment_code(kind) + COMPARE))
        nb['metadata']['casmi_experiment'].update(overlay=kind, measured_score=None,
            validation='Preserved E006 prefix and synthetic paired-overlay tests; Kaggle run pending')
        validate(nb)
        notebooks.append(nb)
    return notebooks


if __name__ == '__main__':
    for name, nb in zip(('casmi26_metric_aligned_simulator_evidence.ipynb',
                         'casmi26_complete_coverage_ice_gl.ipynb'), build()):
        path = ROOT / 'experiments' / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
