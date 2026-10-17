'''Two single-factor follow-ups to the user-confirmed V22 public 0.451 control.'''
import copy
import json
from build_v22_adaptation import build as baseline_build
from build_flow_experiments import ROOT,prose,validate
from build_metric_safe_adaptation import code_cell

FOLLOWUP = '''
class V22Ablation:
    def __init__(self,experiment,sources,mode):
        assert mode in ('no_fragment_gap','promotion_rel200')
        self.scope,self.sources,self.mode = experiment.scope,sources,mode

    def run(self):
        scope = self.scope
        control_bytes = Path('submission.csv').read_bytes()
        Path('submission_v22_0451_control.csv').write_bytes(control_bytes)
        control = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        if self.mode == 'no_fragment_gap':
            # Reconstruct the final overlay from the exact pre-fragment E004 bytes, not from final predictions.
            e004_bytes = Path('submission_e004_control.csv').read_bytes()
            manifest = json.loads(Path('e004_verification.json').read_text())
            assert hashlib.sha256(e004_bytes).hexdigest() == manifest['hashes']['submission.csv']
            Path('submission.csv').write_bytes(e004_bytes)
            scope['FRAG_GAP'] = {}
            V22Stage('Remove only the molecule-level fragment-gap term',[34,36]).run(scope,self.sources)
        else:
            patched = dict(self.sources)
            assert patched[24].count('(rel > 150.0)') == 1
            patched[24] = patched[24].replace('(rel > 150.0)','(rel > 200.0)')
            # Reuse all raw neural predictions; rebuild every descendant of the changed merge.
            V22Stage('rel200 merge and downstream reconstruction',[24,26,29,30,31,33,34,36]).run(scope,patched)
        status = json.loads(Path('e2r2_stats.json').read_text())
        assert status.get('status')=='applied', 'Final metric-exact hedge failed'
        final = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        assert final.columns.tolist() == control.columns.tolist() == ['molecule_id','smiles']
        assert final.molecule_id.tolist() == control.molecule_id.tolist() and final.molecule_id.is_unique
        assert final.smiles.map(lambda s:1 <= len(s.split(';')) <= 25).all()
        inactive = ~control.molecule_id.isin(set(scope['te'].molecule_id.astype(str)))
        assert (final.loc[inactive,'smiles'] == control.loc[inactive,'smiles']).all()
        comparison = pd.DataFrame(dict(molecule_id=control.molecule_id,
            changed=final.smiles!=control.smiles,
            changed_top1=final.smiles.str.split(';').str[0]!=control.smiles.str.split(';').str[0]))
        comparison.to_csv('v22_ablation_comparison.csv',index=False)
        report = dict(mode=self.mode,baseline_user_confirmed_score=.451,measured_score=None,
            relative_threshold=150. if self.mode=='no_fragment_gap' else 200.,
            fragment_gap_enabled=self.mode!='no_fragment_gap',
            changed_rows=int(comparison.changed.sum()),changed_top1=int(comparison.changed_top1.sum()),
            control_sha256=hashlib.sha256(control_bytes).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
            note='Same-run V22 control; source stage audit files after rebuilding describe treatment, not control.')
        Path('v22_ablation_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if not report['changed_rows']:
            print('NO-OP versus V22: do not spend another submission on identical predictions.')
        return report
'''


def build():
    base = baseline_build()[0]
    outputs = []
    for mode,title,method in (
        ('no_fragment_gap','V22 — No Fragment-Gap Ablation',
         'Disable only molecule-level MetFrag-lite fallback where forward-model evidence is absent. '
         'Keep the updated BIO dataset, rel150 promotion, both engines, simulator evidence and '
         'metric-exact rank2 hedge unchanged. Fragment evidence inside engine2 is NOT disabled. '
         'The final list is reconstructed from exact E004 control bytes with FRAG_GAP empty. '
         'This tests the marginal effect of the final fallback, not the usefulness of fragmentation in general.'),
        ('promotion_rel200','V22 — PubChem Promotion rel200',
         'Change only the structure-margin promotion OR branch from rel>150 to rel>200. '
         'Do not change the S>6/pop>=5 clause, gentle/aggressive merge-slot threshold of 600, '
         'fragment-gap policy or metric-exact rank2 hedge. Reuse identical neural/simulator outputs '
         'and rebuild merge/fusion/audit/fragment-gap/E006/hedge descendants. Fragment scores are '
         'recomputed for the actual new candidate groups; there is no second neural inference pass.')):
        nb = copy.deepcopy(base)
        nb['cells'][0] = code_cell(prose('CASMI26: Closing the Evidence Gap\n'+title+'\n'
            'Control: V22 adaptation, public 0.451 confirmed by the user. Treatment score unknown. '
            +method+'\nThis is one isolated change, not a combination with earlier unsuccessful '
            'runner-up, rank3 or supported-hedge experiments. Full author attribution is retained '
            'below. Our addition is the controlled ablation/reconstruction and paired reporting. '
            'A logical hypothesis is not a guarantee of improvement.'))
        nb['cells'].append(code_cell(prose('Independent experiment after the complete V22 control\n'
            +method+'\nSave the byte-exact final control as submission_v22_0451_control.csv. '
            'Inspect v22_ablation_audit.json and v22_ablation_comparison.csv. Earlier '
            'v22_adaptation_audit.json describes the initial control. Intermediate upstream '
            'files can be overwritten by treatment reconstruction, but the control CSV/hash remain '
            'separate. No new datasets: retain the successful V22 inputs, including sergiomalv BIO '
            'and the compatible simulator RDKit wheel for your Python. Full Kaggle GPU execution '
            'and rendered outputs have not been verified locally.')+FOLLOWUP
            +f'ablation = V22Ablation(experiment,SOURCE_CELLS,{mode!r})\nablation_result = ablation.run()\n'))
        nb['metadata']['casmi_experiment'].update(ablation_mode=mode,baseline_user_confirmed_score=.451,
            validation='Synthetic gate boundaries and same-run reconstruction tests; full Kaggle inference pending')
        validate(nb)
        outputs.append(nb)
    return outputs


if __name__=='__main__':
    for name,nb in zip(('casmi26_v22_no_fragment_gap.ipynb','casmi26_v22_promotion_rel200.ipynb'),build()):
        path = ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
