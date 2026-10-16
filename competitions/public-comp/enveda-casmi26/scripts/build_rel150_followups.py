'''Isolated rel150 ablations with a same-run native rank2 control and upstream attribution.'''
import ast
import copy
import json
from build_pubchem_promotion_adaptation import build as base_build, FRAMEWORK
from build_flow_experiments import ROOT, prose, validate
from build_metric_safe_adaptation import code_cell

# Observe immutable strings before the source mutates PubChem candidate ordering.
SNAPSHOT = '''            if index == 29:
                scope['PC_ORIGINAL_LEADERS'] = {str(mid): p['pc'][0] for mid,p in scope['PC'].items() if p.get('pc')}
'''
CAPTURE_FRAMEWORK = FRAMEWORK.replace('        for index in self.indices:\n',
    '        for index in self.indices:\n'+SNAPSHOT)

FOLLOWUPS = '''
class CandidateAlignedPromotion:
    def __init__(self, original_leaders, identity):
        self.original_leaders, self.identity = original_leaders, identity
        self.audit = []

    def allow(self, mid, proposal, popularity_pass):
        old = self.original_leaders.get(str(mid))
        current = proposal.get('pc', [])
        old_key = self.identity(old) if old else None
        current_key = self.identity(current[0]) if current else None
        same = old_key is not None and old_key == current_key
        accepted = bool(popularity_pass and same)
        self.audit.append(dict(molecule_id=str(mid),original_leader=old,
            current_leader=current[0] if current else None,same_metric_identity=same,
            popularity_pass=bool(popularity_pass),aligned_popularity_pass=accepted))
        return accepted

class Rel150Followup:
    def __init__(self, experiment, sources, mode):
        assert mode in ('no_rank2','aligned_promotion')
        self.scope,self.sources,self.mode = experiment.scope,sources,mode

    def run(self):
        scope = self.scope
        control_bytes = Path('submission.csv').read_bytes()
        Path('submission_rel150_rank2_control.csv').write_bytes(control_bytes)
        control = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        if self.mode == 'no_rank2':
            # Native rank2 saved the complete E006 result immediately before its insertion.
            Path('submission.csv').write_bytes(Path('submission_e006.csv').read_bytes())
        else:
            policy = CandidateAlignedPromotion(scope['PC_ORIGINAL_LEADERS'],scope['metric_identity'])
            scope['PROMOTION_ALIGNMENT'] = policy
            patched = dict(self.sources)
            old = '(S is not None and float(S) > S_TAU and pop_top >= POP_TAU)'
            assert patched[32].count(old) == 1
            patched[32] = patched[32].replace(old,'PROMOTION_ALIGNMENT.allow(mid,p,'+old+')')
            # No models or simulator scoring rerun. Rebuild the changed merge and all descendants.
            CreditedStage('Candidate-aligned merge and downstream reconstruction',
                [32,35,39,41,43,45,47]).run(scope,patched)
            pd.DataFrame(policy.audit).to_csv('promotion_alignment_audit.csv',index=False)
        treatment = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        assert treatment.columns.tolist() == control.columns.tolist() == ['molecule_id','smiles']
        assert treatment.molecule_id.is_unique and treatment.molecule_id.tolist() == control.molecule_id.tolist()
        assert treatment.smiles.map(lambda s:1 <= len(s.split(';')) <= 25).all()
        active = set(scope['te'].molecule_id.astype(str))
        inactive = ~control.molecule_id.isin(active)
        assert (control.loc[inactive,'smiles'] == treatment.loc[inactive,'smiles']).all()
        comparison = pd.DataFrame(dict(molecule_id=control.molecule_id,
            changed=control.smiles != treatment.smiles,
            changed_top1=control.smiles.str.split(';').str[0] != treatment.smiles.str.split(';').str[0]))
        if self.mode == 'no_rank2':
            assert not comparison.changed_top1.any()
        comparison.to_csv('rel150_followup_comparison.csv',index=False)
        report = dict(mode=self.mode,relative_threshold=150.,baseline_user_reported_score=.446,
            measured_score=None,changed_rows=int(comparison.changed.sum()),
            changed_top1=int(comparison.changed_top1.sum()),
            control_sha256=hashlib.sha256(control_bytes).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
            note='Same-run native rel150 rank2 control. Source audit files after reconstruction describe treatment; promotion_adaptation_audit.json describes control.')
        if self.mode == 'aligned_promotion':
            report['popularity_clauses_blocked'] = sum(a['popularity_pass'] and not a['same_metric_identity'] for a in policy.audit)
            report['rel_margin_branch_unchanged'] = True
        Path('rel150_followup_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if not report['changed_rows']:
            print('NO-OP versus rel150 rank2 control. Do not spend a submission on identical predictions.')
        return report
'''


def build():
    base = base_build()[1]
    outputs = []
    for mode,title,method in (
        ('no_rank2','Rel150 — No Final Rank-2 Hedge',
         'Remove only the final engine2 rank2 insertion. Engine2 still contributes to retrieval, '
         'metric fusion and the E006 candidate lists. Keep PubChem rel150 and all simulator logic. '
         'Use the byte-exact same-run submission_e006.csv as treatment. This tests the marginal '
         'effect of the last hedge, not whether engine2 itself is useful. Top1 must be unchanged. '
         'Correct engine2 leaders can lose reciprocal-rank credit; improvement is not guaranteed.'),
        ('aligned_promotion','Rel150 — Candidate-Aligned PubChem Promotion',
         'Before simulator reranking, record the original PubChem leader associated with S and top_pop. '
         'At the merge allow the S>6 and pop>=5 clause only when the current leader shares that '
         'original leader official metric identity. Compare with the unchanged upstream metric_identity '
         'function; a different SMILES of the same identity is allowed. Missing/invalid identity blocks '
         'only this popularity clause. The rel>150 OR branch is deliberately unchanged, including '
         'the existing no-pool-candidate sentinel. Ordinary rank2 remains active. No score is recomputed '
         'or attributed to another candidate. The source confidence may be useful at list level, '
         'so this stricter attribution can lower MRR.')):
        nb = copy.deepcopy(base)
        nb['cells'][0] = code_cell(prose('CASMI26: Closing the Evidence Gap\n'+title+'\n'
            'Independent single-factor treatment on rel150 + native rank2, public 0.446 confirmed '
            'by the user. This new version has no measured score. '+method+'\n'
            'All author links, original source credits and licence boundaries remain below. '
            'Our additions are the isolated ablation/alignment policy and paired audits, not the '
            'original promotion, predictors or rank2 idea. Same dataset versions as rel150; no new inputs. '
            'Python3.13 still requires the compatible RDKit2025 cp313 simulator wheel in addition to '
            'the main RDKit2026 wheel. Full GPU execution and rendered presentation remain unverified locally.'))
        source = ''.join(nb['cells'][3]['source'])
        assert source.count(FRAMEWORK) == 1
        nb['cells'][3] = code_cell(source.replace(FRAMEWORK,CAPTURE_FRAMEWORK))
        nb['cells'].append(code_cell(prose('Independent rel150 follow-up\n'+method+'\n'
            'Run the complete unchanged numerical rel150 recipe and source/OOP hedge parity first. '
            'Save its exact final bytes as submission_rel150_rank2_control.csv. '
            'For aligned promotion rerun only deterministic merge/fusion/audit descendants using '
            'the SAME model and simulator outputs. Existing intermediate source audit files may '
            'be overwritten by that treatment; the native control CSV and its hash remain separate. '
            'Inspect rel150_followup_audit.json, rel150_followup_comparison.csv and, when applicable, '
            'promotion_alignment_audit.csv. Smoke no-op does not prove hidden inactivity. '
            'Do not stack failed hedge or fragment experiments.')+FOLLOWUPS
            +f'followup = Rel150Followup(experiment,SOURCE_CELLS,{mode!r})\nfollowup_result = followup.run()\n'))
        nb['metadata']['casmi_experiment'].update(followup_mode=mode,baseline_user_reported_score=.446,
            validation='Synthetic alignment truth table and paired pipeline tests; full Kaggle execution pending')
        validate(nb)
        outputs.append(nb)
    return outputs


if __name__=='__main__':
    for name,nb in zip(('casmi26_rel150_no_rank2.ipynb','casmi26_rel150_aligned_promotion.ipynb'),build()):
        path = ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
