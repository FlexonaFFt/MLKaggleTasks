'''Two independent single-factor hedge tests after the source-verified rel100 recipe.'''
import copy
import json
from build_pubchem_promotion_adaptation import build as base_build
from build_flow_experiments import ROOT, prose, validate
from build_metric_safe_adaptation import code_cell

HEDGES = '''
class RankThreeHedge:
    def __init__(self, key):
        self.key = key

    def rank(self, tokens, engine):
        if not tokens or not engine:
            return list(tokens)
        candidate = engine[0]
        identity = self.key(candidate)
        # Never demote an existing first- or second-place candidate to third.
        if identity is None or identity in {self.key(s) for s in tokens[:2]}:
            return list(tokens)
        result, seen = [], set()
        for smile in tokens[:2]+[candidate]+tokens[2:]:
            key = self.key(smile) or smile
            if key not in seen:
                seen.add(key)
                result.append(smile)
        return result[:25]

class SupportedRankTwoHedge:
    def __init__(self, key, identity):
        self.original = EngineTwoHedge(key)
        self.identity = identity

    def supported(self, engine, first):
        if not engine:
            return False
        identity = self.identity(engine[0])
        if identity is None:
            return False
        # Physical positions of the pre-fusion list, not a new compacted/unique ranking.
        first = [s for s in first if s and s != 'CCO'][:10]
        return identity in {self.identity(s) for s in first}

    def rank(self, tokens, engine, first):
        return self.original.rank(tokens,engine) if self.supported(engine,first) else list(tokens)

class HedgeExperiment:
    def __init__(self, experiment, mode):
        assert mode in ('rank3','supported_rank2')
        self.scope, self.mode = experiment.scope, mode

    def run(self):
        scope = self.scope
        # The parent experiment already executed and checked the exact source rank2 oracle.
        control_bytes = Path('submission.csv').read_bytes()
        Path('submission_rel100_rank2_control.csv').write_bytes(control_bytes)
        control = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        before = pd.read_csv('submission_e006.csv',dtype=str,keep_default_na=False)
        assert control.columns.tolist() == before.columns.tolist() == ['molecule_id','smiles']
        assert control.molecule_id.tolist() == before.molecule_id.tolist()
        assert before.molecule_id.is_unique
        active = set(scope['te'].molecule_id.astype(str))
        third = RankThreeHedge(scope['_keyp'])
        supported = SupportedRankTwoHedge(scope['_keyp'],scope['metric_identity'])
        rows, audits = [], []
        for row,reference in zip(before.itertuples(index=False),control.itertuples(index=False)):
            mid, tokens = row.molecule_id,row.smiles.split(';')
            revised, support = list(tokens),False
            if mid in active:
                engine = scope['ENG'].get(mid,{})
                candidates = [s for s in (engine.get('smiles') or []) if s and s != 'CCO']
                first = scope['PREFUSION_LISTS'].get(mid,[])
                if self.mode == 'rank3':
                    revised = third.rank(tokens,candidates)
                    assert revised[:2] == tokens[:2], 'Rank3 hedge moved original top2'
                else:
                    support = supported.supported(candidates,first)
                    revised = supported.rank(tokens,candidates,first)
                assert revised[:1] == tokens[:1], 'Hedge moved original top1'
                assert all(scope['_keyp'](s) is not None for s in revised)
            else:
                assert row.smiles == reference.smiles, 'Inactive source control changed'
            assert 1 <= len(revised) <= 25
            value = ';'.join(revised)
            rows.append((mid,value))
            audits.append(dict(molecule_id=mid,active=mid in active,supported_top10=support,
                changed_vs_e006=value != row.smiles,changed_vs_rank2=value != reference.smiles,
                changed_top1=revised[:1] != reference.smiles.split(';')[:1]))
        frame = pd.DataFrame(rows,columns=before.columns)
        assert frame.molecule_id.tolist() == control.molecule_id.tolist()
        assert not any(a['changed_top1'] for a in audits)
        frame.to_csv('submission.csv',index=False)
        pd.DataFrame(audits).to_csv('hedge_comparison.csv',index=False)
        report = dict(mode=self.mode,relative_threshold=100.,measured_score=None,
            changed_rows_vs_rank2=sum(a['changed_vs_rank2'] for a in audits),
            changed_rows_vs_e006=sum(a['changed_vs_e006'] for a in audits),
            changed_top1=0,supported_active_rows=sum(a['active'] and a['supported_top10'] for a in audits),
            control_sha256=hashlib.sha256(control_bytes).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
            note='Same-run rel100 rank2 control; no independently measured score. Active molecules only.')
        Path('hedge_experiment_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if not report['changed_rows_vs_rank2']:
            print('NO-OP versus same-run rel100 rank2: do not submit an identical prediction file.')
        return report
'''


def build():
    base = base_build()[0]
    outputs = []
    for mode,title,method in (
        ('rank3','Engine-2 Rank-3 Hedge',
         'Insert the engine2 leader at rank3 instead of rank2, preserving the E006 first two candidates. '
         'If already in the first two positions by the source key, do not demote it. The rest uses '
         'the source hedge key/deduplication semantics. Hypothesis: weaker promotion protects useful '
         'second-place answers. Risk: a correct engine2 answer receives only 1/3 rather than 1/2 MRR.'),
        ('supported_rank2','Top-10 Supported Engine-2 Hedge',
         'Apply the original rank2 hedge only if the engine2 leader official metric identity occurs '
         'in the first ten non-placeholder entries of PREFUSION_LISTS, the engine1-side list AFTER '
         'PubChem merge/promotion and BEFORE fusion. This is not the raw engine1 list. Compare using '
         'the existing metric_identity function, not raw SMILES or the source score_key. Retain the '
         'source hedge/deduplication arithmetic on accepted rows; otherwise retain E006. '
         'Hypothesis: cross-list support suppresses wrong hedge promotions. Risk: reject unique '
         'correct engine2 discoveries. Support can include PubChem evidence and is not independent validation.')):
        nb = copy.deepcopy(base)
        nb['cells'][0] = code_cell(prose('CASMI26: Closing the Evidence Gap\n'+title+'\n'
            'Independent single-factor experiment on the rel100 E006 recipe, not a confirmed new baseline. '
            +method+'\nDo not stack rel150, fragment or simulator experiments. Same inputs as rel100; no new datasets. '
            'All original credits and source links remain in the following cell. Our contribution is '
            'the hedge policy and paired audit; the public rank2 idea belongs to @antriksharya. '
            'No public/private score improvement is claimed.'))
        nb['cells'].append(code_cell(prose('Isolated hedge experiment and paired comparison\n'+method+'\n'
            'The unchanged parent recipe first produces and verifies its native rank2 output. '
            'Save those exact bytes as submission_rel100_rank2_control.csv. Build this treatment '
            'from submission_e006.csv, NOT from the rank2 output, to avoid double insertion. '
            'Inspect hedge_experiment_audit.json and hedge_comparison.csv; initial promotion audits '
            'still describe the control. Zero changed rows means no new prediction to test. '
            'Full GPU inference and rendered notebook inspection require Kaggle and remain pending.')
            +HEDGES+f'hedge_experiment = HedgeExperiment(experiment,{mode!r})\n'
            +'hedge_result = hedge_experiment.run()\n'))
        nb['metadata']['casmi_experiment'].update(hedge_mode=mode,
            validation='Synthetic hedge and paired file tests; full Kaggle inference pending')
        validate(nb)
        outputs.append(nb)
    return outputs


if __name__ == '__main__':
    for name,nb in zip(('casmi26_rel100_rank3_hedge.ipynb','casmi26_rel100_supported_rank2_hedge.ipynb'),build()):
        path = ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
